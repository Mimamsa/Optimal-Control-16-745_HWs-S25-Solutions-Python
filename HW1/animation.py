from math import sin, cos, pi
import numpy as np
import meshcat
import meshcat.geometry as mg
import meshcat.transformations as mtf


def meshcat_animate(params, X, dt, N):
    vis = meshcat.Visualizer()

    rod1 = mg.Cylinder(height=params.L1, radius=0.005)
    rod2 = mg.Cylinder(height=params.L2, radius=0.005)
    vis['rod1'].set_object(rod1)
    vis['rod2'].set_object(rod2)

    sphere = mg.Sphere(radius=0.1)
    vis['s1'].set_object(sphere)
    vis['s2'].set_object(sphere)

    anim = meshcat.animation.Animation(default_framerate=int(1/dt))

    camera_path = "/Cameras/default/rotated/<object>"

    for k in range(0, N):
        with anim.at_frame(vis, k) as frame:
            frame[camera_path].set_property("zoom", "number", 0.5)

            theta_1 = X[k][0]
            theta_2 = X[k][2]
            r1 = np.array([0, params.L1*sin(theta_1), -params.L1*cos(theta_1) + 2])
            r2 = r1 + np.array([0, params.L2*sin(theta_2), -params.L2*cos(theta_2)])
            frame['s1'].set_transform(mtf.translation_matrix(r1))
            frame['s2'].set_transform(mtf.translation_matrix(r2))
            frame['rod1'].set_transform(mtf.compose_matrix(translate=0.5*(np.array([0,0,2]) + r1), angles=np.array([theta_1 + pi/2, 0, 0])))
            frame['rod2'].set_transform(mtf.compose_matrix(translate=r1 + 0.5*(r2 - r1), angles=np.array([theta_2 + pi/2, 0, 0])))

    vis.set_animation(anim)
    vis.open()
    #vis.jupyter_cell()

