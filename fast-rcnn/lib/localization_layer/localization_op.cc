#include <stdio.h>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framewor/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Localization")
  .Attr("T: {float, double}")
  .Attr("output_height: int")
  .Attr("output_width: int")
  .Attr("spatial_scale: float")
  .Input("bottom_data: T")
  .Input("bottom_rois: T")
  .Output("top_data: T")
  .Output("argmax: int32");

REGISTER_OP("LocalizationGrad")
  .Attr("T: {float, double}")
  .Attr("output_height: int")
  .Attr("output_width: int")
  .Attr("spatial_scale: float")
  .Input("bottom_data: T")
  .Input("bottom_rois: T")
  .Input("argmax: int32")
  .Input("grad: T")
  .Output("output: T")

template <typename Device, typename T>
class LocalizationOp : public OpKernel {
  public:
    explicit LocalizationOp(OpKernelConstruction* context) : OpKernel(context) {
      // Get the output height
      OP_REQUIRES_OK(context, 
                    context->GetAttr("output_height", &output_height_));
      // Check that output_height is positive
      OP_REQUIRES(context, output_height_ >= 0,
                errors::InvalidArgument("Need output_height >= 0, got ", output_height_));
      // Get the output width
      OP_REQUIRES_OK(context, 
                    context->GetAttr("output_width", &output_width_));
      // Check that output_width is positive
      OP_REQUIRES(context, output_width_ >= 0,
                errors::InvalidArgument("Need output_width >= 0, got ", output_width_));

      // Get the spatial scale
      OP_REQUIRES_OK(context,
                    context->GetAttr("spatial_scale", &spatial_scale_))
    }

    void Compute(OpKernelContext* context) override {
      // Get the input tensor
      const Tensor& bottom_data = context->input(0);
      const Tensor& bottom_rois = context->input(1);
      auto bottom_data_flat = bottom_data.flat<T>();
      auto bottom_rois_flat = bottom_rois.flat<T>();

      // data should have 4 dimensions
      // 4-tensor (batch_size, H, W, C)
      OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

      // rois should have 2 dimensions
      // 2-tensor (num_rois, [x1, y1, x2, y2])
      OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

      // Number of ROIs
      int num_rois = bottom_rois.dim_size(0);
      
      // batch_size
      int batch_size = bottom_data.dim_size(0);
      // data height
      int data_height = bottom_data.dim_size(1);
      // data width
      int data_width = bottom_data.dim_size(2);
      // number of channels
      int num_channels = bottom_data.dim_size(3);

      // construct the output shape
      int dims[4];
      dims[0] = num_rois;
      dims[1] = output_height_;
      dims[2] = output_width_;
      dims[3] = num_channels;
      TensorShape output_shape;
      TensorShapeUtils::MakeShape(dims, 4, &output_shape);

      // Create output tensors
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
      auto output = output_tensor->template flat<T>();

      Tensor* argmax_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &argmax_tensor));
      auto argmax = argmax_tensor->template flat<int>();

      // Set all elements of the output tensor to -inf
      const int N = output.size();
      for (int i = 0; i < N; i++) {
        output(i) = -FLT_MAX;
        argmax(i) = -1;
      }

      // For each ROI R = [batch_index x1 y1 x2 y2]: bilinear sample over R
      int index_roi = 0;
      int index_output = 0;
      for (int n = 0; n < num_rois; ++n) {
        int roi_batch_ind = bottom_rois_flat(index_roi + 0);
        int roi_start_w = round(bottom_rois_flat(index_roi + 1) * spatial_scale_);
        int roi_start_h = round(bottom_rois_flat(index_roi + 2) * spatial_scale_);
        int roi_end_w = round(bottom_rois_flat(index_roi + 3) * spatial_scale_);
        int roi_end_h = round(bottom_rois_flat(index_roi + 4) * spatial_scale_);
        CHECK_GE(roi_batch_ind, 0);
        CHECK_LT(roi_batch_ind, batch_size);

        int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
        int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
        //const T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(output_height_);
        //const T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(output_width_);

        // Construct affine matrix A
        // expects input vector [y, x, 1] instead of [x, y 1]
        const T A11 = static_cast<T>(roi_height) / static_cast<T>(data_height);
        const T A13 = static_cast<T>(roi_start_h + roi_end_h - data_height - 1) / static_cast<T>(data_height - 1);
        const T A22 = static_cast<T>(roi_width) / static_cast<T>(data_width);
        const T A23 = static_cast<T>(roi_start_w + roi_end_w - data_width - 1) / static_cast<T>(data_width - 1);

        int index_data = roi_batch_ind * data_height * data_width * num_channels;

        for (int oh = 0; oh < output_height_; ++oh) {
          for (int ow = 0; ow < output_width_; ++ow) {
            // Compute sampled region for this output unit
            // **************
            // output of matmul with A should be normalized coords in range [-1, 1]
            T norm_y = oh * A11 + A13;
            T norm_x = ow * A22 + A23;
            // NOTE: taken from 
            // https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py
            T y = (norm_y + 1.0) * (data_height) / 2.0;
            T x = (norm_x + 1.0) * (data_width) / 2.0;
            for (int h = 0; h < data_height; h++) {
              for (int w = 0; w < data_width; w++) {
                for (int c = 0; c < num_channels; c++) {
                  const int input_index = index_data + (h * data_width + w) * num_channels + c;
                  const int output_index = index_output + (oh * output_width + w) * num_channels + c;
                  output(output_index) = bottom_data_flat(input_index) * 
                    bilinear_kernel(w - x) * bilinear_kernel(h - y);
                }
              }
            }

          }
        }
        // Increment ROI index
        index_roi += bottom_rois.dim_size(1);
        index_output += output_height_ * output_width_ * num_channels;
      }

    }

  private:
    int output_height_;
    int output_width;
    float spatial_scale_;

    double bilinear_kernel(double x){
      return std::max(0, 1 - std::fabs(x));
    }

};

REGISTER_KERNEL_BUILDER(Name("Localization").Device(DEVICE_CPU).TypeConstraint<float>("T"), LocalizationOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Localization").Device(DEVICE_CPU).TypeConstraint<double>("T"), LocalizationOp<CPUDevice, double>);

bool LocalizationForwardLauncher(
  const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
  const int width, const int channels, const int output_height,
  const int output_width, const float* bottom_rois,
  float* top_data, int* argmax_data, const Eigen::GpuDevice& d);

static void LocalizationKernel(
  OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_rois,
  const float spatial_scale, const int num_rois, const int height,
  const int width, const int channels, const int output_height,
  const int output_width, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  Tensor* argmax = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape, &argmax));

  if (!context->status().ok()) { return; }

  LocalizationForwardLauncher(
    bottom_data->flat<float>().data(), spatial_scale, num_rois, height,
    width, channels, output_height, output_width, bottom_rois->flat<float>().data(),
    output->flat<float>().data(), argmax->flat<int>().data(), context->eigen_device<Eigen::GpuDevice>()
  );
}

template <class T>
class LocalizationOp<Eigen::GpuDevice, T> : public OpKernel {
  public:
    typedef Eigen::GpuDevice Device;

    explicit LocalizationOp(OpKernelConstruction* context) : OpKernel(context) {
      
      // Get the output height
      OP_REQUIRES_OK(context, context->GetAttr("output_height", &output_height_));
      // Check that output height is positive
      OP_REQUIRES(context, output_height_ >= 0,
        errors::InvalidArgument("Need output_height >= 0, got ", output_height_));

      // Get the output width
      OP_REQUIRES_OK(context, context->GetAttr("output_width", &output_width_));
      // Check that output_width is positive
      OP_REQUIRES(context, output_width_ >= 0,
        errors::InvalidArgument("Need pooled_width >= 0, got ", output_width_));
      // Get the spatial scale
      OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale_));
    }

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensor
      const Tensor& bottom_data = context->input(0);
      const Tensor& bottom_rois = context->input(1);

      // data should have 4 dimensions
      OP_REQUIRES(context, bottom_data.dims() == 4,
        errors::InvalidArgument("data must be 4-dimensional"));
      // rois should have 2 dimensions
      OP_REQUIRES(context, bottom_rois.dims() == 2,
        errors::InvalidArgument("rois must be 2-dimensional"));

      // Number of ROIs
      int num_rois = bottom_rois.dim_size(0);
      // batch size
      int batch_size = bottom_data.dim_size(0);
      // data height
      int data_height = bottom_data.dim_size(1);
      // data width
      int data_width = bottom_data.dim_size(2);
      // Number of channels
      int num_channels = bottom_data.dim_size(3);

      // construct the output shape
      int dims[4];
      dims[0] = num_rois;
      dims[1] = output_height_;
      dims[2] = output_width_;
      dims[3] = num_channels;
      TensorShape output_shape;
      TensorShapeUtils::MakeShape(dims, 4, &output_shape);

      LocalizationPoolingKernel(context, &bottom_data, &bottom_rois, spatial_scale_, num_rois, data_height,
        data_width, num_channels, output_height_, output_width, output_shape);
    }

  private:
    int output_height_;
    int output_width;
    float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("Localization").Device(DEVICE_GPU).TypeConstraint<float>("T"), LocalizationOp<Eigen::GpuDevice, float>);

bool LocalizationBackwardLauncher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
  const int height, const int width, const int channels, const int output_height,
  const int output_width, const float* bottom_rois, float* bottom_diff, const int* argmax_data, const Eigen::GpuDevice& d);

static void LocalizationGradKernel(
  OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_rois, const Tensor* argmax_data, const Tensor* out_backprop,
  const float spatial_scale, const int batch_size, cont int num_rois, const int height, const int width, const int channels,
  const int output_height, const int output_width, const TensorShape& tensor_output_shape)
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) { return; }

  LocalizationBackwardLauncher(
    output_backprop->flat<float>().data(), spatial_scale, batch_size, num_rois, height, width, channels,
    output_height, output_width, bottom_rois->flat<float>().data(), output->flat<float>().data(),
    argmax_data->flat<int>().data(), context->eigen_device<Eigen::GpuDevice>()
  );
}

// compute gradient
template <class Device, class T>
class LocalizationGradOp : public OpKernel {
  public:
    explicit LocalizationGradOp(OpKernelConstruction* context) : OpKernel(context) {
      
      // Get the output height
      OP_REQUIRES_OK(context, context->GetAttr("output_height", &output_height_));
      // Check that output_height is positive
      OP_REQUIRES(context, output_height_ >= 0,
        errors::InvalidArgument("Need output_height >= 0, got ", output_height_));
      // Get the output width
      OP_REQUIRES_OK(context, context->GetAttr("output_width", &output_width_));
      // Check that output_width is positive
      OP_REQUIRES(context, output_width_ >= 0,
        errors::InvalidArgument("Need output_width >= 0, got ", output_width_));
      // Get the spatial scale
      OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale_));
    }

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensor
      const Tensor& bottom_data = context->input(0);
      const Tensor& bottom_rois = context->input(1);
      const Tensor& argmax_data = context->input(2);
      const Tensor& out_backprop = context->input(3);

      // data should have 4 dimensions
      OP_REQUIRES(context, bottom_data.dims() == 4,
        errors::InvalidArgument("data must be 4-dimensional"));
      // rois should have 2 dimensions
      OP_REQUIRES(context, bottom_rois.dims() == 2,
        errors::InvalidArgument("rois must be 2-dimensional"));
      OP_REQUIRES(context, argmax_data.dims() == 4,
        errors::InvalidArgument("argmax_data must be 4-dimensional"));
      OP_REQUIRES(context, out_backprop.dims() == 4,
        errors::InvalidArgument("out_backprop must be 4-dimensional"));

      // Number of ROIs
      int num_rois = bottom_rois.dim_size(0);
      // batch size
      int batch_size = bottom_data.dim_size(0);
      // data height
      int height = bottom_data.dim_size(1);
      // data width
      int width = bottom_data.dim_size(2);
      // Number of channels
      int channels = bottom_data.dim_size(3);

      // construct the output shape
      TensorShape output_shape = bottom_data.shape();

      LocalizationGradKernel(
        context, &bottom_data, &bottom_rois, &argmax_data, &out_backprop,
        spatial_scale_, batch_size, num_rois, height, width, channels, output_height_,
        output_width_, output_shape);
    }

  private:
    int output_height_;
    int output_width;
    float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("LocalizationGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
  LocalizationGradOp<Eigen::GpuDevice, float>);
