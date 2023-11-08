use std::{sync::Arc, path::Path, vec};
use ndarray::{Array, ArrayD, IxDyn, ArrayView2, Array2, Ix, Axis, ArrayView3, s};
use anyhow::{anyhow,Result}; // For the anyhow! macro
use opencv::{
    prelude::*,
    videoio,
    highgui
}; // Note, the namespace of OpenCV is changed (to better or worse). It is no longer one enormous.
use ort::{Environment,SessionBuilder,Value};
use image::{ImageBuffer, ImageOutputFormat};
use image::{GenericImageView, imageops::FilterType};
use image::RgbImage;





fn main() -> Result<()> { // Note, this is anyhow::Result
    // Open a GUI window
    highgui::named_window("window", highgui::WINDOW_FULLSCREEN)?;
    // Open the web-camera (assuming you have one)
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let mut frame = Mat::default(); // This array will store the web-cam data
    // Read the camera
    // and display in the window
	let env = Arc::new(Environment::builder().with_name("YOLOv8").build().unwrap());
    let model = SessionBuilder::new(&env).unwrap().with_model_from_file("yolov8m.onnx").unwrap();
    loop {
        cam.read(&mut frame)?;
        highgui::imshow("window", &frame)?;
        let key = highgui::wait_key(1)?;
        if key == 113 { // quit with q
            break;
        }
	if !frame.is_continuous(){
		return Err(anyhow!("Frame is not continuous!"));
	}
	let data_bytes = frame.data_bytes()?;
    let img = image::load_from_memory(&data_bytes)?;
    println!("Image Loaded! {}", img.width());
    // let (img_width, img_height) = (img.width(), img.height());
    // let img = img.resize_exact(640, 640, FilterType::CatmullRom);
    // let mut input = Array::zeros((1, 3, 640, 640)).into_dyn();
    // for pixel in img.pixels() {
    //     let x = pixel.0 as usize;
    //     let y = pixel.1 as usize;
    //     let [r,g,b,_] = pixel.2.0;
    //     input[[0, 0, y, x]] = (r as f32) / 255.0;
    //     input[[0, 1, y, x]] = (g as f32) / 255.0;
    //     input[[0, 2, y, x]] = (b as f32) / 255.0;
    // };
    // let input_as_values = &input.as_standard_layout();
    // let model_inputs = vec![Value::from_array(model.allocator(), input_as_values).unwrap()];
    // //let outputs = model.run(model_inputs).unwrap();
}
Ok(())
}


fn iou(box1: &(f32, f32, f32, f32, &'static str, f32), box2: &(f32, f32, f32, f32, &'static str, f32)) -> f32 {
    return intersection(box1, box2) / union(box1, box2);
}

// Function calculates union area of two boxes
// Returns Area of the boxes union as a float number
fn union(box1: &(f32, f32, f32, f32, &'static str, f32), box2: &(f32, f32, f32, f32, &'static str, f32)) -> f32 {
    let (box1_x1,box1_y1,box1_x2,box1_y2,_,_) = *box1;
    let (box2_x1,box2_y1,box2_x2,box2_y2,_,_) = *box2;
    let box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1);
    let box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1);
    return box1_area + box2_area - intersection(box1, box2);
}

// Function calculates intersection area of two boxes
// Returns Area of intersection of the boxes as a float number
fn intersection(box1: &(f32, f32, f32, f32, &'static str, f32), box2: &(f32, f32, f32, f32, &'static str, f32)) -> f32 {
    let (box1_x1,box1_y1,box1_x2,box1_y2,_,_) = *box1;
    let (box2_x1,box2_y1,box2_x2,box2_y2,_,_) = *box2;
    let x1 = box1_x1.max(box2_x1);
    let y1 = box1_y1.max(box2_y1);
    let x2 = box1_x2.min(box2_x2);
    let y2 = box1_y2.min(box2_y2);
    return (x2-x1)*(y2-y1);
}

// fn convert_vec_f32_to_image(vec_f32: Vec<f32>, width: u32, height: u32, color_type: image::ColorType) -> ImageBuffer<image::Rgb<u8>, Container> {
//     let mut img_buf = ImageBuffer::new(width, height, color_type);
//     for (y, row) in vec_f32.chunks_exact(width as usize).enumerate() {
//         for (x, pixel) in row.iter().enumerate() {
//             let pixel_u8 = (*pixel * 255.0) as u8;
//             let pixel_rgb = image::Rgb([pixel_u8, pixel_u8, pixel_u8]);
//             img_buf.put_pixel(x, y, pixel_rgb);
//         }
//     }
//     img_buf
// }

// fn save_image_to_directory(img_buf: ImageBuffer<image::Rgb<u8>, Container>, directory: &str, filename: &str) {
//     std::fs::create_dir_all(directory).unwrap();
//     let full_path = std::path::Path::new(directory).join(filename);
//     img_buf.save(full_path, ImageOutputFormat::Png).unwrap();
// }



// Array of YOLOv8 class labels
const YOLO_CLASSES:[&str;80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];
