use opencv::{highgui, prelude::*, videoio, Result};
use opencv::dnn;
use opencv::core::Scalar;
use opencv::core::CV_32F;
use opencv::core::Size;
use std::fs::File;
use std::io::BufReader;
use opencv::dnn::{read_net_from_onnx, Net, NetTrait};

fn main() -> Result<()> {
	let onnx_file = &"models/yolov8m.onnx";
	let mut model = read_net_from_onnx(onnx_file).expect("Unable to load model!"); 
	let window = "video capture";
	highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;
	let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 is the default camera
	let opened = videoio::VideoCapture::is_opened(&cam)?;
	if !opened {
		panic!("Unable to open default camera!");
	}
	loop {
		let mut frame = Mat::default();
		cam.read(&mut frame)?;
		if frame.size()?.width > 0 {
			highgui::imshow(window, &frame)?;
		}
    	let blob = dnn::blob_from_image(&frame, 1.0, Size::new(416, 416), Scalar::new(127.5, 127.5, 127.5, 127.5), true, false, i32::from(CV_32F))?;     
        let key = highgui::wait_key(10)?;
		if key > 0 && key != 255 {
			break;
		}
	}
	Ok(())
}