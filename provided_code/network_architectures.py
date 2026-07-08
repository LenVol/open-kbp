""" Neural net architectures """
from typing import Optional

import keras
import tensorflow

#from keras.engine.keras_tensor import KerasTensor
from tensorflow.keras import KerasTensor
from keras.layers import Activation, AveragePooling3D, Conv3D, Conv3DTranspose, Input, LeakyReLU, SpatialDropout3D, concatenate, MaxPooling3D, UpSampling3D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import Optimizer

from provided_code.data_shapes import DataShapes


class DefineDoseFromCT:
    """This class defines the architecture for a U-NET and must be inherited by a child class that
    executes various functions like training or predicting"""

    def __init__(
        self,
        data_shapes: DataShapes,
        initial_number_of_filters: int,
        filter_size: tuple[int, int, int],
        stride_size: tuple[int, int, int],
        gen_optimizer: Optimizer,
    ):
        self.data_shapes = data_shapes
        self.initial_number_of_filters = initial_number_of_filters
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.gen_optimizer = gen_optimizer

    def make_convolution_block(self, x: KerasTensor, num_filters: int, strides: tuple=(1,1,1), use_batch_norm: bool = True) -> KerasTensor:
        x = Conv3D(num_filters, self.filter_size, strides=strides, padding="same", use_bias=False)(x)
        if use_batch_norm:
            x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        return x


    # Dense convolution block as in Wu et al. 2021 "Improving Proton Dose Calculation Accuracy by Using Deep Learning"
    def make_dense_convolution_block(self, x: KerasTensor, num_filters: int) -> KerasTensor:
        y = Conv3D(num_filters, self.filter_size, strides=(1,1,1), padding="same", use_bias=False)(x)
        y = BatchNormalization(momentum=0.99, epsilon=1e-3)(y)
        y = LeakyReLU(negative_slope=0.2)(y)
        x = concatenate([x,y])
        return x

    def make_dense_pool(self, x: KerasTensor, num_filters: int) -> KerasTensor:
        y = MaxPooling3D(pool_size=(2,2,2))(x)
        z = Conv3D(num_filters, self.filter_size, strides=(2,2,2), padding="same", use_bias=False)(x)
        z = BatchNormalization(momentum=0.99, epsilon=1e-3)(z)
        z = LeakyReLU(negative_slope=0.2)(z)
        x = concatenate([y,z])
        return x

    def make_dense_upsampling(self, x: KerasTensor, num_filters: int) -> KerasTensor:
        x = UpSampling3D(size=(2,2,2))(x)
        x = Conv3D(num_filters, self.filter_size, strides=(1,1,1), padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        return x
    

    def make_convolution_transpose_block(
        self, x: KerasTensor, num_filters: int, use_dropout: bool = True, skip_x: Optional[KerasTensor] = None
    ) -> KerasTensor:
        if skip_x is not None:
            x = concatenate([x, skip_x])
        x = Conv3DTranspose(num_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        if use_dropout:
            x = SpatialDropout3D(0.2)(x)
        x = LeakyReLU(negative_slope=0)(x)  # Use LeakyReLU(alpha = 0) instead of ReLU because ReLU is buggy when saved
        return x

    def define_generator_old(self) -> Model:
        """Makes a generator that takes a CT image as input to generate a dose distribution of the same dimensions"""

        # Define inputs
        ct_image = Input(self.data_shapes.ct)
        roi_masks = Input(self.data_shapes.structure_masks)

        # Build Model starting with Conv3D layers
        x = concatenate([ct_image, roi_masks])
        x1 = self.make_convolution_block(x, self.initial_number_of_filters)
        x2 = self.make_convolution_block(x1, 2 * self.initial_number_of_filters)
        x3 = self.make_convolution_block(x2, 4 * self.initial_number_of_filters)
        x4 = self.make_convolution_block(x3, 8 * self.initial_number_of_filters)
        x5 = self.make_convolution_block(x4, 8 * self.initial_number_of_filters)
        x6 = self.make_convolution_block(x5, 8 * self.initial_number_of_filters, use_batch_norm=False)

        # Build model back up from bottleneck
        x5b = self.make_convolution_transpose_block(x6, 8 * self.initial_number_of_filters, use_dropout=False)
        x4b = self.make_convolution_transpose_block(x5b, 8 * self.initial_number_of_filters, skip_x=x5)
        x3b = self.make_convolution_transpose_block(x4b, 4 * self.initial_number_of_filters, use_dropout=False, skip_x=x4)
        x2b = self.make_convolution_transpose_block(x3b, 2 * self.initial_number_of_filters, skip_x=x3)
        x1b = self.make_convolution_transpose_block(x2b, self.initial_number_of_filters, use_dropout=False, skip_x=x2)

        # Final layer
        x0b = concatenate([x1b, x1])
        x0b = Conv3DTranspose(1, self.filter_size, strides=self.stride_size, padding="same")(x0b)
        x_final = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding="same")(x0b)
        final_dose = Activation("relu")(x_final)

        # Compile model for use
        generator = Model(inputs=[ct_image, roi_masks], outputs=final_dose, name="generator")
        generator.compile(loss="mean_absolute_error", optimizer=self.gen_optimizer)
        generator.summary()
        return generator
###################################################################################################
# Define new generator based on HD-UNet, see Huet et al. 2023. https://arxiv.org/pdf/2310.19686, and 
# Wu et al. 2021 "Improving Proton Dose Calculation Accuracy by Using Deep Learning"
    def define_generator(self) -> Model:
        # Define inputs
        ct_image = Input(self.data_shapes.ct)
        print(ct_image.shape)
        roi_masks = Input(self.data_shapes.structure_masks)
        x = concatenate([ct_image,roi_masks])
    
        num_filters = 1
        ## encoder shape 128
        x1 = self.make_dense_convolution_block(x, num_filters)
        x1b = self.make_convolution_block(x1, num_filters)
        x1c = self.make_dense_pool(x1b, num_filters)


        # shape 64
        x2 = self.make_dense_convolution_block(x1c, num_filters)
        x2b = self.make_dense_convolution_block(x2, num_filters)
        x2c = self.make_dense_pool(x2b, num_filters)

        #shape 32
        x3 = self.make_dense_convolution_block(x2c, num_filters)
        x3b = self.make_dense_convolution_block(x3, num_filters)
        x3c = self.make_dense_pool(x3b, num_filters)
        
        #shape 16
        x4 = self.make_dense_convolution_block(x3c, num_filters)
        x4b = self.make_dense_convolution_block(x4, num_filters)
        x4c = self.make_dense_pool(x4b, num_filters)

        ## Bottleneck shape 8 
        x5 = self.make_dense_convolution_block(x4c, num_filters)
        x5b = self.make_dense_convolution_block(x5, num_filters)


        # decoder 
        print(x5b.shape)
        x6 = self.make_dense_upsampling(x5b, num_filters)
        print(x6.shape)
        x6b = concatenate([x4b,x6])
        x6c = self.make_dense_convolution_block(x6b, num_filters)
        x6d = self.make_dense_convolution_block(x6c, num_filters)
        print(x6d.shape)
       
        x7 = self.make_dense_upsampling(x6d, num_filters)
        print(x7.shape)
        x7b = concatenate([x3b,x7])
        x7c = self.make_dense_convolution_block(x7b, num_filters)
        x7d = self.make_dense_convolution_block(x7c, num_filters)

        x8 = self.make_dense_upsampling(x7d, num_filters)
        x8b = concatenate([x2b,x8])
        x8c = self.make_dense_convolution_block(x8b, num_filters)
        x8d = self.make_dense_convolution_block(x8c, num_filters)

        x9 = self.make_dense_upsampling(x8d, num_filters)
        x9b = concatenate([x1b,x9])
        x9c = self.make_dense_convolution_block(x9b, num_filters)
        x9d = self.make_dense_convolution_block(x9c, num_filters)

        x_final = Conv3D(num_filters, self.filter_size, strides=(1,1,1), padding="same", use_bias=False)(x9d)

        final_dose = Activation("relu")(x_final)
        
        # Compile model for use
        generator = Model(inputs=[ct_image, roi_masks], outputs=final_dose, name="generator")
        generator.compile(loss="mean_absolute_error", optimizer=self.gen_optimizer)
        generator.summary()
        return generator

