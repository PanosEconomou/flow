import taichi as ti

ti.init(arch = ti.gpu)

n = 1000
pixels = ti.Vector.field(4,ti.f32, (n*2,n))

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1]*z[0]*2])

@ti.kernel
def paint(t: float):
    for i,j in pixels:
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i/n - 1, j/n - 0.5])*2
        iterations = 0
        while z.norm() < 20 and iterations < 100:
            z = complex_sqr(z) + c
            iterations += 1
        pixels [i,j] = [1 - iterations*0.02,0,0,1]

gui = ti.GUI("JULIA SET!", res = (n*2,n), fast_gui=True)


for i in range(10000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()