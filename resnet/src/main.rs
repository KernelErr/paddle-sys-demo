use paddle_sys;
use libc::{c_void, calloc, free, malloc};
use std::ffi::{CStr, CString};
use std::mem::{size_of, transmute};
use std::ptr::write;

fn main() {
    #[cfg(not(target_os = "linux"))]
    const LIB_PATH: &str = "paddle_fluid_c.dll";

    #[cfg(target_os = "linux")]
    const LIB_PATH: &str = "libpaddle_fluid_c.so";

    unsafe {
        let paddle = paddle_sys::bindings::paddle_fluid_c::new(LIB_PATH).unwrap();
        let config = paddle.PD_NewAnalysisConfig();

        let model_path = CString::new("/path/to/model").unwrap();
        let param_path = CString::new("/path/to/params").unwrap();
        paddle.PD_SetModel(config, model_path.as_ptr(), param_path.as_ptr());

        let input_tensor = paddle.PD_NewPaddleTensor();

        let input_buffer = paddle.PD_NewPaddleBuf();
        match paddle.PD_PaddleBufEmpty(input_buffer) {
            true => println!("PaddleBuf empty"),
            false => panic!("PaddleBuf NOT empty"),
        }

        let batch = 1;
        let channel = 3;
        let height = 318;
        let width = 318;
        let input_shape: [i32; 4] = [batch, channel, height, width];
        let input_size: usize = (batch * channel * height * width) as usize;
        let input_data = calloc(size_of::<f32>(), input_size) as *mut f32;
        for i in 0..input_size {
            let root: *mut f32 =
                transmute(transmute::<*mut f32, u64>(input_data) + (size_of::<f32>() * i) as u64);
            write(root, 1.0);
        }
        paddle.PD_PaddleBufReset(
            input_buffer,
            input_data as *mut c_void,
            (size_of::<f32>() * input_size) as u64,
        );

        let input_name = CString::new("data").unwrap();
        paddle.PD_SetPaddleTensorName(input_tensor, input_name.as_ptr() as *mut i8);
        paddle.PD_SetPaddleTensorDType(input_tensor, paddle_sys::bindings::PD_DataType_PD_FLOAT32);
        paddle.PD_SetPaddleTensorShape(input_tensor, input_shape.as_ptr() as *mut i32, 4);
        paddle.PD_SetPaddleTensorData(input_tensor, input_buffer);

        let output_tensor = &mut paddle.PD_NewPaddleTensor();
        let output_size: *mut i32 = malloc(size_of::<i32>()) as *mut i32;
        paddle.PD_PredictorRun(config, input_tensor, 1, output_tensor, output_size, 1);

        println!("Output Tensor Size: {}", &*output_size);
        println!(
            "Output Tensor Name: {}",
            CStr::from_ptr(paddle.PD_GetPaddleTensorName(*output_tensor))
                .to_str()
                .unwrap()
        );
        println!(
            "Output Tensor Dtype: {}",
            paddle.PD_GetPaddleTensorDType(*output_tensor)
        );

        let output_buffer = paddle.PD_GetPaddleTensorData(*output_tensor);
        let result_length = paddle.PD_PaddleBufLength(output_buffer) as usize / size_of::<f32>();
        println!("Output Data Length: {}", result_length);
        assert_eq!(result_length, 512);

        free(input_data as *mut c_void);
        free(output_size as *mut c_void);
        paddle.PD_DeletePaddleTensor(input_tensor);
        paddle.PD_DeletePaddleBuf(input_buffer);
        paddle.PD_DeleteAnalysisConfig(config);
    }
}
