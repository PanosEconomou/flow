from numpy import pi, linspace                                  # Numpy helpers
import taichi as ti                                             # GPU Accelleration
import taichi.math as tm                                        # It's math module
ti.init(arch = ti.gpu, default_fp=ti.f32, fast_math=True)       # Start the engine

## SOLVE THE PENDULUM --------------------------------------------------------------------

# Derivative function
@ti.func
def f(u, a, b):
    return tm.vec2([u.y,-a*u.y -b*ti.sin(u.x)])

# Solve the rk step
@ti.func
def step_u(u, a:float, b:float, h:float=1e-1):
    k1 = f(u,        a,b)
    # k2 = f(u+k1*h/2, a,b)
    # k3 = f(u+k2*h/2, a,b)
    # k4 = f(u+k3*h,   a,b)

    return u + h*(k1)# + 2*k2 + 2*k3 + k4)/6

# Given a starting point in phase space, find where it ends
@ti.func
def evolve(u0, a:float, b:float, h:float, max_iter:int=1500, threshold:float=1e-3):
    u       = u0
    i       = 0
    # run     = True
    # while run:
    #     u   = step_u(u, a, b, h)    # Step along the path
    #     i+=1
    #     run = (i<max_iter) and ((ti.sin(u.x)*ti.sin(u.x)) + (u.y*u.y) > threshold )
    for k in range(max_iter):
        u   = step_u(u, a, b, h)    # Step along the path
        if (ti.sin(u.x)*ti.sin(u.x)) + (u.y*u.y) > threshold :
            i = k
    return tm.vec3(u.x,u.y,i)


## SETUP THE VISUAL --------------------------------------------------------------------

# The picture
frames  = 200                                   # Number of frames
n       = 500                                   # Number of Pixels
L       = 3*pi                                  # Length to look at
pixels  = ti.Vector.field(3,ti.f32, (2*n,n))    # Stores the color of each pixel
window  = ti.ui.Window("2D Waves", res=(2*n, n), fps_limit=400)
gui = window.get_canvas()
# gui     = ti.GUI("A Phase space", res = (2*n,n), fast_gui=True) # type: ignore

# A colormap
@ti.func
def colormap(u,L):
    val  = (tm.clamp(u.x, -2*L, 2*L) + 2*L)/(4*L)
    return tm.vec3([val, (1 - ti.min(u.z,1500)/1500) , val])
    
# Color each pixel
@ti.kernel
def paint(a:float, b:float, L:float, n:int):
    for i,j in pixels:
        u0 = tm.vec2(2*i*L/(2*n) - L,L*j/n - L/2)
        u  = evolve(u0, a, b, 1e-2)
        pixels[i,j] = colormap(u,L)

# Now draw the animation
A = linspace(0.0001,1,frames)
i = 0
b = 2
paused = False

while window.running:    
    paint(A[int(abs(i - frames + 1))],2,L,n)
    i = (i + 1)%(2*(frames-1))
    gui.set_image(pixels)
    window.show()