use std::{f32::consts::PI, num::NonZeroU32};

use femtovg::{Canvas, Color, ImageFlags, ImageId, Paint, Path, RenderTarget, renderer::OpenGl};
use glutin::{
    config::ConfigTemplateBuilder,
    context::{ContextAttributesBuilder, PossiblyCurrentContext},
    display::GetGlDisplay,
    prelude::*,
    surface::{Surface, WindowSurface},
};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasWindowHandle;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

//display settings
const ZOOM_SCALE: f32 = 300.0;
const DISPLAY_OFFSET_X: f32 = ZOOM_SCALE / 3.0;
const DISPLAY_OFFSET_Y: f32 = ZOOM_SCALE / 3.0;
const DISPLAY_WIDTH: f32 = ZOOM_SCALE / 150.0 * 400.0;
const DISPLAY_HEIGHT: f32 = ZOOM_SCALE / 150.0 * 250.0;
const FTLE_DISPLAY_COLOR_MAX: f32 = 0.5;

//particle settings
const AMOUNT_OF_PARTICLES: i32 = 1000;
const CENTER_SPAWN_X: f32 = 1.0;
const X_AMPLITUDE: f32 = 1.0;
const CENTER_SPAWN_Y: f32 = 0.5;
const Y_AMPLITUDE: f32 = 0.5;

//ftle settings
const GRID_COLS: f32 = 80.0;
const GRID_ROWS: f32 = GRID_COLS / 2.0;
const GRID_CALC_T0: f32 = 0.0;
const GRID_CALC_TF: f32 = 10.0;

const DYNAMIC_FTLE: bool = true;
const FTLE_PERIOD: f32 = 10.0; //depende da função fluxo que estamos utilizando
const TIME_PER_FTLE: f32 = 0.1;

//simulation settings
const TIME_BETWEEN_SAMPLES: f32 = 0.1; //in seconds

#[derive(Clone, Copy, Debug)]
struct Particle {
    x: f32,
    y: f32,
}
#[derive(Clone, Copy, Debug)]
struct FtleCell {
    screen_x: f32,
    screen_y: f32,
    cell_w: f32,
    cell_h: f32,
    color: (u8, u8, u8),
}

impl Default for Particle {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

struct AppState {
    window: Window,
    gl_surface: Surface<WindowSurface>,
    gl_context: PossiblyCurrentContext,
    canvas: Canvas<OpenGl>,
    color: u8,
    particles: Vec<Particle>,
    current_time: f32,
    ftle_cells_imgs: Vec<femtovg::ImageId>,
}

#[derive(Default)]
struct App {
    state: Option<AppState>,
}

fn velocity_field(x: f32, y: f32, t: f32) -> (f32, f32) {
    //receives position and time, returns velocity

    let epsilon = 0.25;
    let a = 0.1;
    let omega = 2.0 * PI / 10.0;

    let f =
        epsilon * ((omega * t).sin()) * x.powi(2) + (1.0 - 2.0 * epsilon * ((omega * t).sin())) * x;
    let delf_del_x =
        epsilon * ((omega * t).sin()) * 2.0 * x - 2.0 * epsilon * ((omega * t).sin()) + 1.0;

    return (
        -PI * a * ((PI * f).sin()) * ((PI * y).cos()),
        PI * a * ((PI * f).cos()) * ((PI * y).sin()) * delf_del_x,
    );
}

fn rk4_advect(mut x: f32, mut y: f32, mut t: f32, t_end: f32, steps: usize) -> (f32, f32) {

    if steps == 0 || (t_end - t).abs() < 1e-6 {
        return (x, y);
    }
    let dt = (t_end - t) / steps as f32;

    for _ in 0..steps {
        let (k1_x, k1_y) = velocity_field(x, y, t);
        let (k2_x, k2_y) = velocity_field(x + 0.5 * dt * k1_x, y + 0.5 * dt * k1_y, t + 0.5 * dt);
        let (k3_x, k3_y) = velocity_field(x + 0.5 * dt * k2_x, y + 0.5 * dt * k2_y, t + 0.5 * dt);
        let (k4_x, k4_y) = velocity_field(x + dt * k3_x, y + dt * k3_y, t + dt);

        x += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);
        y += (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y);
        t += dt;
    }

    (x, y)
}

fn deformation_gradient(x0: f32, y0: f32, t0: f32, t: f32) -> [f32; 4] {
    let delta = 1e-4; // perturbação infinitesimal controlada
    let steps = ((t - t0).abs() * 20.0).max(10.0) as usize;

    let (px_plus, py_plus) = rk4_advect(x0 + delta, y0, t0, t, steps); //+delta no x
    let (px_minus, py_minus) = rk4_advect(x0 - delta, y0, t0, t, steps); //-delta no x
    let (qx_plus, qy_plus) = rk4_advect(x0, y0 + delta, t0, t, steps); //+delta no y
    let (qx_minus, qy_minus) = rk4_advect(x0, y0 - delta, t0, t, steps);//-delta no y

    let two_delta = 2.0 * delta;

    [
        (px_plus - px_minus) / two_delta, // dF1/dx0
        (qx_plus - qx_minus) / two_delta, // dF1/dy0
        (py_plus - py_minus) / two_delta, // dF2/dx0
        (qy_plus - qy_minus) / two_delta, // dF2/dy0
    ]
}

fn cauchy_green_tensor(deformation_matrix: [f32; 4]) -> [f32; 4] {
    let mut result = [f32::default(); 4];

    result[0] = deformation_matrix[0] * deformation_matrix[0]
        + deformation_matrix[2] * deformation_matrix[2];
    result[1] = deformation_matrix[0] * deformation_matrix[1]
        + deformation_matrix[2] * deformation_matrix[3];
    result[2] = deformation_matrix[1] * deformation_matrix[0]
        + deformation_matrix[3] * deformation_matrix[2];
    result[3] = deformation_matrix[1] * deformation_matrix[1]
        + deformation_matrix[3] * deformation_matrix[3];

    return result;
}

fn compute_ftle(cauchy_green: [f32; 4], dt: f32) -> f32 {
    let lambda2 = (cauchy_green[0]
        + cauchy_green[3]
        + ((cauchy_green[0] - cauchy_green[3]).powi(2) + 4.0 * cauchy_green[1].powi(2)).sqrt())
        / 2.0;
    if lambda2 <= 1.0 || dt.abs() < 1e-4 {
        return 0.0;
    }
    return (0.5 / dt.abs()) * lambda2.ln();
}

fn generate_initial_particles(amount: i32) -> Vec<Particle> {
    let mut particles = vec![Particle::default(); amount as usize];

    let initial_x = CENTER_SPAWN_X - X_AMPLITUDE;
    let final_x = CENTER_SPAWN_X + X_AMPLITUDE;

    let initial_y = CENTER_SPAWN_Y - Y_AMPLITUDE;
    let final_y = CENTER_SPAWN_Y + Y_AMPLITUDE;

    let mut increment_x = 0.0;
    let mut increment_y = 0.0;

    let distance_between_particles =
        ((final_x - initial_x) * (final_y - initial_y) / amount as f32).sqrt();

    for particle in &mut particles {
        particle.x = initial_x + increment_x as f32;
        increment_x += distance_between_particles;
        particle.y = initial_y + increment_y as f32;
        if initial_x + increment_x > final_x {
            increment_x = 0.0;
            increment_y += distance_between_particles;
        }
    }

    return particles;
}

fn turbo_heat_map(t: f32, max: f32) -> (u8, u8, u8) {
    // Clamp t to [0, 1]
    let t = t / max;

    // Approximate component curves
    let r = (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0);

    return (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    );
}

fn precompute_ftle_grid(t0: f32, t_f: f32) -> Vec<FtleCell> {
    let x_min = 0.0;
    let x_max = 2.0;
    let y_min = 0.0;
    let y_max = 1.0;

    let dx = (x_max - x_min) / GRID_COLS as f32;
    let dy = (y_max - y_min) / GRID_ROWS as f32;
    
    let mut cells = vec![FtleCell{screen_x:0.0, screen_y:0.0, cell_w:0.0, cell_h:0.0, color:(0,0,0)}; (GRID_ROWS*GRID_COLS) as usize];

    for row in 0..GRID_ROWS as i32 {
        let y = y_min + (row as f32 + 0.5) * dy;
        for col in 0..GRID_COLS as i32 {
            let x = x_min + (col as f32 + 0.5) * dx;

            let grad = deformation_gradient(x, y, t0, t_f);
            let cg = cauchy_green_tensor(grad);
            let ftle = compute_ftle(cg, t_f - t0);

            cells[(row*GRID_COLS as i32 + col) as usize] = FtleCell {
                screen_x: x * ZOOM_SCALE + DISPLAY_OFFSET_X,
                screen_y: y * ZOOM_SCALE + DISPLAY_OFFSET_Y,
                cell_w: dx * ZOOM_SCALE,
                cell_h: dy * ZOOM_SCALE,
                color: turbo_heat_map(ftle, FTLE_DISPLAY_COLOR_MAX),
            };
            
        }
    }
    cells
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attributes = WindowAttributes::default()
            .with_title("Iniciação Científica - 2026")
            .with_inner_size(LogicalSize::new(DISPLAY_WIDTH, DISPLAY_HEIGHT));

        let template = ConfigTemplateBuilder::new();
        let display_builder = DisplayBuilder::new().with_window_attributes(Some(window_attributes));

        let (window, gl_config) = display_builder
            .build(event_loop, template, |configs| {
                configs.into_iter().next().unwrap()
            })
            .expect("Failed to build GL display and window");

        let window = window.expect("Failed to create window");
        let gl_display = gl_config.display();

        // Window handle extraction
        let raw_window_handle = window
            .window_handle()
            .expect("Failed to get window handle")
            .as_raw();

        let context_attributes = ContextAttributesBuilder::new().build(Some(raw_window_handle));
        let not_current_gl_context = unsafe {
            gl_display
                .create_context(&gl_config, &context_attributes)
                .expect("Failed to create GL context")
        };

        let surface_attributes = window
            .build_surface_attributes(Default::default())
            .expect("Failed to build surface attributes");

        let gl_surface = unsafe {
            gl_display
                .create_window_surface(&gl_config, &surface_attributes)
                .expect("Failed to create window surface")
        };

        let gl_context = not_current_gl_context
            .make_current(&gl_surface)
            .expect("Failed to make GL context current");

        let renderer = unsafe {
            OpenGl::new_from_function_cstr(|s| gl_display.get_proc_address(s) as *const _)
                .expect("Cannot create OpenGl renderer")
        };

        let mut canvas = Canvas::new(renderer).expect("Cannot create Canvas");

        let cache_width = DISPLAY_WIDTH as usize;
        let cache_height = DISPLAY_HEIGHT as usize;

        let mut cached_img_ids: Vec<ImageId> = vec![];

        if DYNAMIC_FTLE {

            let mut every_grid_ftle_cells = vec![vec![FtleCell{screen_x:0.0, screen_y:0.0, cell_w:0.0, cell_h:0.0, color:(0,0,0)}; (GRID_ROWS * GRID_COLS) as usize]; (FTLE_PERIOD / TIME_PER_FTLE) as usize];

            every_grid_ftle_cells.par_iter_mut().enumerate().for_each(|(i, ftle_cell_grid)|{
                (*ftle_cell_grid) = precompute_ftle_grid(
                    GRID_CALC_T0 + TIME_PER_FTLE * i as f32,
                    GRID_CALC_TF + TIME_PER_FTLE * i as f32,
                );
                println!("precomputed the {} of {} ftle!", i, FTLE_PERIOD / TIME_PER_FTLE);
            });

            for i in 0..(FTLE_PERIOD / TIME_PER_FTLE) as i32 {
                let cache_image_id = canvas
                    .create_image_empty(
                        cache_width,
                        cache_height,
                        femtovg::PixelFormat::Rgba8,
                        ImageFlags::empty(), // or ImageFlags::empty()
                    )
                    .expect("Failed to create offscreen image");

                // 1. Direct drawing to the off-screen image
                canvas.set_render_target(RenderTarget::Image(cache_image_id));

                // 2. Clear the target
                canvas.clear_rect(
                    0,
                    0,
                    cache_width as u32,
                    cache_height as u32,
                    femtovg::Color::rgba(0, 0, 0, 0),
                );

                // 3. Draw ftle static squares

                let ftle_cells = &every_grid_ftle_cells[i as usize];

                for ftle_cell in ftle_cells {
                    let mut square = Path::new();
                    square.rect(
                        ftle_cell.screen_x,
                        ftle_cell.screen_y,
                        ftle_cell.cell_w,
                        ftle_cell.cell_h,
                    );
                    let rgb_values = ftle_cell.color;
                    let stroke_paint =
                        Paint::color(Color::rgb(rgb_values.0, rgb_values.1, rgb_values.2));
                    canvas.fill_path(&square, &stroke_paint);
                }
                cached_img_ids.push(cache_image_id);
                println!(
                    "ftle image {} of {} calculated!",
                    i + 1,
                    FTLE_PERIOD as f32 / TIME_PER_FTLE
                );
            }
        } else {
            let cache_image_id = canvas
                .create_image_empty(
                    cache_width,
                    cache_height,
                    femtovg::PixelFormat::Rgba8,
                    ImageFlags::empty(), // or ImageFlags::empty()
                )
                .expect("Failed to create offscreen image");

            // 1. Direct drawing to the off-screen image
            canvas.set_render_target(RenderTarget::Image(cache_image_id));

            // 2. Clear the target
            canvas.clear_rect(
                0,
                0,
                cache_width as u32,
                cache_height as u32,
                femtovg::Color::rgba(0, 0, 0, 0),
            );

            // 3. Draw ftle static squares

            let ftle_cells = precompute_ftle_grid(GRID_CALC_T0, GRID_CALC_TF);

            for ftle_cell in ftle_cells {
                let mut square = Path::new();
                square.rect(
                    ftle_cell.screen_x,
                    ftle_cell.screen_y,
                    ftle_cell.cell_w,
                    ftle_cell.cell_h,
                );
                let rgb_values = ftle_cell.color;
                let stroke_paint =
                    Paint::color(Color::rgb(rgb_values.0, rgb_values.1, rgb_values.2));
                canvas.fill_path(&square, &stroke_paint);
            }
            cached_img_ids.push(cache_image_id);
        }

        // 4. Switch back to the main display
        canvas.set_render_target(RenderTarget::Screen);

        self.state = Some(AppState {
            window,
            gl_surface,
            gl_context,
            canvas,
            color: 0,
            particles: generate_initial_particles(AMOUNT_OF_PARTICLES),
            ftle_cells_imgs: cached_img_ids,
            current_time: 0.0,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.gl_surface.resize(
                        &state.gl_context,
                        NonZeroU32::new(new_size.width).unwrap(),
                        NonZeroU32::new(new_size.height).unwrap(),
                    );
                }
            }
            WindowEvent::RedrawRequested => {
                let dt = TIME_BETWEEN_SAMPLES;
                state.current_time += dt;

                let size = state.window.inner_size();
                let scale = state.window.scale_factor() as f32;

                state.canvas.set_size(size.width, size.height, scale);
                state
                    .canvas
                    .clear_rect(0, 0, size.width, size.height, Color::rgb(150, 150, 150));

                // 2. Draw particles and update position
                let ftle_img_paint;
                if DYNAMIC_FTLE {
                    ftle_img_paint = Paint::image(
                        state.ftle_cells_imgs
                            [(state.current_time % FTLE_PERIOD).div_euclid(TIME_PER_FTLE) as usize],
                        0.0,
                        0.0,
                        DISPLAY_WIDTH,
                        DISPLAY_HEIGHT,
                        0.0,
                        1.0,
                    );
                } else {
                    ftle_img_paint = Paint::image(
                        state.ftle_cells_imgs[0],
                        0.0,
                        0.0,
                        DISPLAY_WIDTH,
                        DISPLAY_HEIGHT,
                        0.0,
                        1.0,
                    );
                }

                let mut ftle_rect = Path::new();
                ftle_rect.rect(0.0, 0.0, DISPLAY_WIDTH, DISPLAY_HEIGHT);
                state.canvas.fill_path(&ftle_rect, &ftle_img_paint);

                for particle in &mut state.particles {
                    let mut circle = Path::new();
                    circle.circle(
                        particle.x * ZOOM_SCALE + DISPLAY_OFFSET_X,
                        particle.y * ZOOM_SCALE + DISPLAY_OFFSET_Y,
                        0.001,
                    );
                    // let mut stroke_paint = Paint::color(Color::rgb((particle.x / 2.0 * 255.0) as u8, (particle.y / 1.0 * 255.0) as u8, 125));
                    let mut stroke_paint = Paint::color(Color::rgb(255, 255, 255));
                    stroke_paint.set_line_width(5.0);
                    state.canvas.stroke_path(&mut circle, &stroke_paint);

                    let (vx, vy) =
                        velocity_field(particle.x, particle.y, state.current_time as f32);
                    particle.x += vx * dt;
                    particle.y += vy * dt;
                }

                state.color = state.color.wrapping_add(1);

                state.canvas.flush();
                state.gl_surface.swap_buffers(&state.gl_context).unwrap();

                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Failed to initialize event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Application crashed");
}
