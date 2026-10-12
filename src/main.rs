use std::{f32::consts::PI, num::NonZeroU32, time::Instant};

use femtovg::{Canvas, Color, Paint, Path, renderer::OpenGl};
use glutin::{
    config::ConfigTemplateBuilder,
    context::{ContextAttributesBuilder, PossiblyCurrentContext},
    display::GetGlDisplay,
    prelude::*,
    surface::{Surface, WindowSurface},
};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasWindowHandle;
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

//particle settings
const AMOUNT_OF_PARTICLES: i32 = 50000;
const CENTER_SPAWN_X: f32 = 1.0;
const X_AMPLITUDE: f32 = 1.0;
const CENTER_SPAWN_Y: f32 = 0.5;
const Y_AMPLITUDE: f32 = 0.5;

//ftle settings
const GRID_COLS: f32 = 50.0;
const GRID_ROWS: f32 = GRID_COLS/2.0;
const GRID_CALC_T0: f32 = 0.0;
const GRID_CALC_TF: f32 = 10.0;

//simulation settings
const TIME_BETWEEN_SAMPLES: f32 = 0.1; //in seconds

#[derive(Clone, Copy, Debug)]
struct Particle {
    x: f32,
    y: f32,
}
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
    last_frame: Instant,
    app_start: Instant,
    ftle_cells: Vec<FtleCell>,
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
    /* minha versão que era meio bosta
    let initial_x = CENTER_SPAWN_X - X_AMPLITUDE;
    let final_x = CENTER_SPAWN_X + X_AMPLITUDE;

    let initial_y = CENTER_SPAWN_Y - Y_AMPLITUDE;
    let final_y = CENTER_SPAWN_Y + Y_AMPLITUDE;

    let mut result = [f32::default(); 4];

    let delta = ((final_x-initial_x)*(final_y-initial_y)/AMOUNT_OF_GRADIENT_CELLS as f32).sqrt();

    let dt = t - t0;

    let vel0 = velocity_field(x0 + delta, y0, t0);
    let vel1 = velocity_field(x0 - delta, y0, t0);
    let vel2 = velocity_field(x0, y0 + delta, t0);
    let vel3 = velocity_field(x0, y0 - delta, t0);

    result[0] = (x0 + vel0.0 * dt - (x0 + vel1.0 * dt)) / (2.0 * delta.abs());
    result[1] = (x0 + vel2.0 * dt - (x0 + vel3.0 * dt)) / (2.0 * delta.abs());
    result[2] = (y0 + vel0.1 * dt - (y0 + vel1.1 * dt)) / (2.0 * delta.abs());
    result[3] = (y0 + vel2.1* dt - (y0 + vel3.1 * dt)) / (2.0 * delta.abs());

    return result;
    */
    let delta = 1e-4; // perturbação infinitesimal controlada
    let steps = ((t - t0).abs() * 20.0).max(10.0) as usize;

    let (px_plus, py_plus) = rk4_advect(x0 + delta, y0, t0, t, steps);
    let (px_minus, py_minus) = rk4_advect(x0 - delta, y0, t0, t, steps);
    let (qx_plus, qy_plus) = rk4_advect(x0, y0 + delta, t0, t, steps);
    let (qx_minus, qy_minus) = rk4_advect(x0, y0 - delta, t0, t, steps);

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

    let mut values = vec![0.0; (GRID_COLS * GRID_ROWS) as usize];
    let mut max_val: f32 = 0.0;

    let mut counter = 0;
    for row in 0..GRID_ROWS as i32 {
        let y = y_min + (row as f32 + 0.5) * dy;
        for col in 0..GRID_COLS as i32 {
            let x = x_min + (col as f32 + 0.5) * dx;

            let grad = deformation_gradient(x, y, t0, t_f);
            let cg = cauchy_green_tensor(grad);
            let ftle = compute_ftle(cg, t_f - t0);

            let idx = row * GRID_COLS as i32 + col;
            values[idx as usize] = ftle;
            if ftle > max_val {
                max_val = ftle;
            }
            counter += 1;

            if counter % 1000 == 0 {
                println!("precompute process is now {:.2}% done", counter as f32/(GRID_COLS*GRID_ROWS)*100.0);
            }
        }
    }

    println!("FTLE concluído. Maior valor encontrado: {:.4}", max_val);

    let mut cells = Vec::with_capacity((GRID_COLS * GRID_ROWS) as usize);

    for row in 0..(GRID_ROWS as i32) {
        let y = y_min + row as f32 * dy;
        for col in 0..(GRID_COLS as i32) {
            let x = x_min + col as f32 * dx;
            let val = values[(row * GRID_COLS as i32 + col) as usize];

            cells.push(FtleCell {
                screen_x: x * ZOOM_SCALE + DISPLAY_OFFSET_X,
                screen_y: y * ZOOM_SCALE + DISPLAY_OFFSET_Y,
                cell_w: dx * ZOOM_SCALE,
                cell_h: dy * ZOOM_SCALE,
                color: turbo_heat_map(val, max_val),
            });
        }
    }

    cells
}
/*
fn draw_deformation_gradient(canvas: &mut Canvas<OpenGl>, t: f32, t0: f32) {
    let initial_x = -0.25;
    let final_x = 2.25;

    let initial_y = -0.25;
    let final_y = 1.25;

    let mut increment_x = 0.0;
    let mut increment_y = 0.0;

    let distance_between_particles =
        ((final_x - initial_x) * (final_y - initial_y) / AMOUNT_OF_GRADIENT_CELLS as f32).sqrt();

    let mut deformation_values = vec![f32::default(); AMOUNT_OF_GRADIENT_CELLS as usize];
    let mut biggest_value: f32 = 0.0;

    for value in &mut deformation_values {
        let x0 = initial_x + increment_x as f32;
        increment_x += distance_between_particles;
        let y0 = initial_y + increment_y as f32;
        if initial_x + increment_x > final_x {
            increment_x = 0.0;
            increment_y += distance_between_particles;
        }

        (*value) = compute_ftle(
            cauchy_green_tensor(deformation_gradient(x0, y0, t0, t)),
            t - t0,
        );
        if *value > biggest_value {
            biggest_value = *value;
        }
    }

    increment_x = 0.0;
    increment_y = 0.0;

    let mut pos_x = initial_x;
    let mut pos_y = initial_y;
    for deformation_value in deformation_values {
        let mut square = Path::new();
        square.rect(
            (pos_x - distance_between_particles / 2.0) * ZOOM_SCALE + DISPLAY_OFFSET_X,
            (pos_y - distance_between_particles / 2.0) * ZOOM_SCALE + DISPLAY_OFFSET_Y,
            distance_between_particles * ZOOM_SCALE,
            distance_between_particles * ZOOM_SCALE,
        );
        let rgb_values = turbo_heat_map(deformation_value, biggest_value);
        let stroke_paint = Paint::color(Color::rgb(rgb_values.0, rgb_values.1, rgb_values.2));
        canvas.fill_path(&square, &stroke_paint);

        pos_x = initial_x + increment_x as f32;
        increment_x += distance_between_particles;
        pos_y = initial_y + increment_y as f32;
        if initial_x + increment_x > final_x {
            increment_x = 0.0;
            increment_y += distance_between_particles;
        }
    }
}
*/
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

        let canvas = Canvas::new(renderer).expect("Cannot create Canvas");

        self.state = Some(AppState {
            window,
            gl_surface,
            gl_context,
            canvas,
            color: 0,
            particles: generate_initial_particles(AMOUNT_OF_PARTICLES),
            last_frame: Instant::now(),
            app_start: Instant::now(),
            ftle_cells: precompute_ftle_grid(GRID_CALC_T0, GRID_CALC_TF),
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
                
                // descomentar para intervalos contínuos durante a execução
                // let now = Instant::now();
                // let dt = (now - state.last_frame).as_secs_f32();
                // state.last_frame = now;


                let size = state.window.inner_size();
                let scale = state.window.scale_factor() as f32;

                state.canvas.set_size(size.width, size.height, scale);
                state
                    .canvas
                    .clear_rect(0, 0, size.width, size.height, Color::rgb(150, 150, 150));

                // 2. Draw particles and update position
                //draw_deformation_gradient(&mut state.canvas, state.app_start.elapsed().as_secs_f32(), 0.0);
                for ftle_cell in &mut state.ftle_cells {
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
                        state.canvas.fill_path(&square, &stroke_paint);
                    
                }
                

                for particle in &mut state.particles {
                    let mut circle = Path::new();
                    circle.circle(particle.x * ZOOM_SCALE + DISPLAY_OFFSET_X, particle.y * ZOOM_SCALE + DISPLAY_OFFSET_Y, 0.001);
                    // let mut stroke_paint = Paint::color(Color::rgb((particle.x / 2.0 * 255.0) as u8, (particle.y / 1.0 * 255.0) as u8, 125));
                    let mut stroke_paint = Paint::color(Color::rgb(255, 255, 255));
                    stroke_paint.set_line_width(5.0);
                    state.canvas.stroke_path(&mut circle, &stroke_paint);

                    let (vx, vy) = velocity_field(
                        particle.x,
                        particle.y,
                        state.app_start.elapsed().as_secs_f32(),
                    );
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
