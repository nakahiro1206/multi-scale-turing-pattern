use ndarray::{Array, Array1, Array2, Array3, ArrayBase, OwnedRepr, Dim, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use ndarray_ndimage::{gaussian_filter, BorderMode};
use image::{GrayImage, RgbImage};
use rand::Rng;

// mod fastblur;

// https://softologyblog.wordpress.com/2011/07/05/multi-scale-turing-patterns/

struct Scale {
    activator: ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>, 
    inhibitor: ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>, 
    varidation: ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
    activator_radius: f64, 
    inhibitor_radius: f64, 
    small_amount: f64, 
    weight: f64, 
    color: Array1<u8>
}
impl Scale {
    pub fn new(activator_radius: f64, inhibitor_radius: f64, small_amount: f64, weight: f64, height: usize, width: usize, color: ArrayBase<OwnedRepr<u8>, Dim<[usize; 1]>>) -> Scale{
        Scale {
            activator: Array::<f64, _>::random((height, width), Uniform::new(-1., 1.)), 
            inhibitor: Array::<f64, _>::random((height, width), Uniform::new(-1., 1.)), 
            varidation: Array::<f64, _>::random((height, width), Uniform::new(-1., 1.)),
            activator_radius, 
            inhibitor_radius, 
            small_amount, 
            weight, 
            color,
        }
    }
}

fn load_image(path: String)-> (usize, usize, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>){
    let empty = String::from("");
    if path.eq(&empty) {
        const HEIGHT: usize = 512;
        const WIDTH: usize = 512;

        let grid: ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = Array::<f64, _>::random((HEIGHT, WIDTH), Uniform::new(-1., 1.));
        let colored_grid: ArrayBase<OwnedRepr<u8>, ndarray::Dim<[usize; 3]>> = Array::<u8, _>::zeros((HEIGHT, WIDTH, 3));

        return (HEIGHT, WIDTH, grid, colored_grid);
    }

    let colored_img = image::open(path).unwrap();
    let height = colored_img.height() as usize;
    let width = colored_img.width() as usize;

    let mut gray_img = colored_img.grayscale();
    let gray_img = gray_img.as_mut_luma8().unwrap();
    let gray_arr = gray_img.clone().into_vec();
    let gray_vec = Array::from_shape_vec((height, width), gray_arr).unwrap();
    let gray_vec = gray_vec.mapv(|x| (x as f64) / 128. - 1.);

    let colored_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> = colored_img.to_rgb8();
    let colored_img_arr = colored_img.into_raw();
    let colored_vec = Array::from_shape_vec((height, width, 3), colored_img_arr).unwrap();

    return (height, width, gray_vec, colored_vec);
}

fn pick_random_color(has_input: bool, colored_grid: &ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>) -> ArrayBase<OwnedRepr<u8>, Dim<[usize; 1]>>{
    if ! has_input {
        return Array::<u8, _>::random(3, Uniform::new(0, 255));
    }

    let mut rng = rand::thread_rng();
    let h_r = rng.gen_range(0.0..1.0);
    let w_r = rng.gen_range(0.0..1.0);
    let (height, width, _) = colored_grid.dim();

    let color= colored_grid.slice(s![(height as f64 * h_r) as usize, (width as f64 * w_r) as usize, ..]);
    let color_cloned = color.mapv(|x| x);
    return color_cloned;
}

fn array_to_image(arr: &Array3<u8>) -> RgbImage {
    assert!(arr.is_standard_layout());

    let (height, width, _) = arr.dim();
    let cloned = arr.clone();
    let raw = cloned.into_raw_vec();

    RgbImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}

fn array_to_image_gray(arr: &Array2<f64>) -> GrayImage {
    assert!(arr.is_standard_layout());

    let (height, width) = arr.dim();
    let cloned = arr.clone();
    let cloned_u8 = cloned.mapv(|x|((x+1.) * 128.) as u8);
    let raw = cloned_u8.into_raw_vec();

    GrayImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}

fn main(){
    const SCALE_NUM: usize = 10;
    const STEP_NUM: usize = 100;

    let path = String::from("");
    let has_input = path.ne(&(String::from("")));
    let (height, width, mut grid, mut colored_grid) = load_image(path);
    // let (height, width, mut grid, mut colored_grid) = load_image(String::from(""));

    let gray_image = array_to_image_gray(&grid);
    gray_image.save(format!("img_gray/out_0.png")).unwrap();

    let image = array_to_image(&colored_grid);
    image.save(format!("img/out_0.png")).unwrap();

    let mut scales: Vec<Scale> = vec![];

    // example.
    // scales.push(Scale::new(100., 200., 0.05, 1., HEIGHT, WIDTH));
    // scales.push(Scale::new(20., 40., 0.04, 1., HEIGHT, WIDTH));
    // scales.push(Scale::new(10., 20., 0.03, 1., HEIGHT, WIDTH));
    // scales.push(Scale::new(5., 10., 0.02, 1., HEIGHT, WIDTH));
    // scales.push(Scale::new(1., 2., 0.01, 1., HEIGHT, WIDTH));

    let mut rng = rand::thread_rng();
    let min_height_width = height.min(width) as f64;
    for _scale_idx in 0..SCALE_NUM {
        let r: f64 = rng.gen_range(0.0..1.0);
        let color = pick_random_color(has_input, &colored_grid);
        scales.push(Scale::new(r*min_height_width/4., r * min_height_width/2., r / 25. + 0.01, 1., height, width, color));
    }

    for step in 1..STEP_NUM {
        println!("{step}");

        // https://github.com/johnae/blur
        for scale in &mut scales{

            // fastblur::gaussian_blur(data, width, height, blur_radius);

            // add blur
            scale.activator = scale.weight * gaussian_filter(
                &grid, 
                &scale.activator_radius / 2., 
                0, 
                BorderMode::<f64>::Wrap, // Reflect, 
                3);
            scale.inhibitor = scale.weight * gaussian_filter(
                &grid, 
                &scale.inhibitor_radius / 2., 
                0, 
                BorderMode::<f64>::Wrap, // Reflect, 
                3);
            
            scale.varidation = (&scale.activator - &scale.inhibitor).mapv(|x| x.abs());
        }

        let mut max: f64 = 0.;
        let mut min: f64 = 0.;

        for h in 0..height {
            for w in 0..width {
                let mut best_scale: usize = 0;
                for scale_idx in 0..SCALE_NUM {
                    if scales[scale_idx].varidation[[h, w]] < scales[best_scale].varidation[[h, w]] {
                        best_scale = scale_idx;
                    }
                }

                let mut factor = 0.;
                if scales[best_scale].activator[[h, w]] > scales[best_scale].inhibitor[[h, w]] {
                    factor = scales[best_scale].small_amount;
                }
                else {
                    factor -= scales[best_scale].small_amount;
                }

                // lerp blend.
                let blend_ratio = 0.3;
                colored_grid[[h, w, 0]] = (colored_grid[[h, w, 0]] as f64 * (1. - blend_ratio) + scales[best_scale].color[0] as f64 * blend_ratio) as u8;
                colored_grid[[h, w, 1]] = (colored_grid[[h, w, 1]] as f64 * (1. - blend_ratio) + scales[best_scale].color[1] as f64 * blend_ratio) as u8;
                colored_grid[[h, w, 2]] = (colored_grid[[h, w, 2]] as f64 * (1. - blend_ratio) + scales[best_scale].color[2] as f64 * blend_ratio) as u8;

                // multiply brightness
                colored_grid[[h, w, 0]] = (colored_grid[[h, w, 0]] as f64 * ((grid[[h, w]] + 1.) / 2.)) as u8;
                colored_grid[[h, w, 1]] = (colored_grid[[h, w, 1]] as f64 * ((grid[[h, w]] + 1.) / 2.)) as u8;
                colored_grid[[h, w, 2]] = (colored_grid[[h, w, 2]] as f64 * ((grid[[h, w]] + 1.) / 2.)) as u8;

                grid[[h, w]] += factor;

                max = max.max(grid[[h, w]]);
                min = min.min(grid[[h, w]]);
            }
        }

        let gray_image = array_to_image_gray(&grid);
        gray_image.save(format!("img_gray/out_{step}.png")).unwrap();

        let image = array_to_image(&colored_grid);
        image.save(format!("img/out_{step}.png")).unwrap();

        // Normalization
        grid = grid.mapv(|x| (x - min) / (max - min) * 2. - 1.);
    }

    
}
