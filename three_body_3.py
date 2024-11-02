from manim_vid.manim_imports_ext import *  # type: ignore
from scipy.integrate import solve_ivp


G = 1

total_time = 15

masses = np.array([1, 2, 3])
positions = np.array([[-0.97000436, 0.24308753], [0, 0], [0.97000436, -0.24308753]])
velocities = np.array([[0.4662036850, 0.4323657300], [-0.93240737, -0.86473146], [0.4662036850, 0.4323657300]])
# positions = np.array([[-1, 0.25], [0, 0], [1, -0.25]])
# velocities = np.array([[0.5, 0.5], [-1, -0.9], [0.5, 0.5]])
colors = [TEAL, PURPLE_C, LIGHT_BROWN]


def three_body(t, state, masses, g=G):

    r1, r2, r3 = state[:2], state[2:4], state[4:6]
    v1, v2, v3 = state[6:8], state[8:10], state[10:]

    r12 = r2 - r1
    r13 = r3 - r1
    r23 = r3 - r2
    d12 = np.linalg.norm(r12)
    d13 = np.linalg.norm(r13)
    d23 = np.linalg.norm(r23)

    F12 = g * masses[0] * masses[1] / d12**2 * (r12 / d12)
    F13 = g * masses[0] * masses[2] / d13**2 * (r13 / d13)
    F23 = g * masses[1] * masses[2] / d23**2 * (r23 / d23)

    a1 = (F12 + F13) / masses[0]
    a2 = (-F12 + F23) / masses[1]
    a3 = (-F13 - F23) / masses[2]

    return np.concatenate((v1, v2, v3, a1, a2, a3))


def ode_solution_points(function, state0, time, masses, dt=0.01):
    solution = solve_ivp(function, t_span=(0, time), y0=state0, t_eval=np.arange(0, time, dt), args=(masses,))
    return solution.y.T


class ThreeBody(InteractiveScene):
    def construct(self):

        axes = Axes(
            x_range=(-3, 3, 5),
            y_range=(-3, 3, 5),
        )
        axes.set_width(FRAME_WIDTH)
        axes.center()

        equations = Tex(
            f"""
            \\begin{{aligned}}
            &\\mathrm{{Initial\\ conditions\\ (arb.\\ units)}}\\\\
            \\\\
            &G = {G}\\\\
            \\\\
            &\\mathrm{{Body\\ 1}}:\\\\
            &m = {masses[0]}\\\\
            &r = ({positions[0][0]}, {positions[0][1]})\\\\
            &v = ({velocities[0][0]}, {velocities[0][1]})\\\\
                \\\\
            &\\mathrm{{Body\\ 2}}:\\\\
            &m = {masses[1]}\\\\
            &r = ({positions[1][0]}, {positions[1][1]})\\\\
            &v = ({velocities[1][0]}, {velocities[1][1]})\\\\
                \\\\
            &\\mathrm{{Body\\ 3}}:\\\\
            &m = {masses[2]}\\\\
            &r = ({positions[2][0]}, {positions[2][1]})\\\\
            &v = ({velocities[2][0]}, {velocities[2][1]})
            \\end{{aligned}}
            """,
            t2c={
                "Body\\ 1": colors[0],
                "Body\\ 2": colors[1],
                "Body\\ 3": colors[2],
            },
            font_size=15,
        )
        equations.fix_in_frame()
        equations.to_corner()
        # equations.set_backstroke()
        self.add(equations)

        initial_state = np.concatenate((positions.flatten(), velocities.flatten()))

        curves = VGroup()
        points = ode_solution_points(three_body, initial_state, total_time, masses=masses)

        dots = Group(Dot(fill_color=colors[i], radius=0.1 * masses[i]) for i in range(3))

        for body in range(3):
            curve = VMobject().set_points_smoothly(axes.c2p(points[:, 2 * body].T, points[:, 2 * body + 1].T))
            curve.set_stroke(colors[body], width=2, opacity=0.5)
            curves.add(curve)

        def update_dots(dots, curves=curves):
            for dot, curve in zip(dots, curves):
                dot.move_to(curve.get_end())

        dots.add_updater(update_dots)

        tail = VGroup(TracingTail(dot, time_traced=1, stroke_width=5).match_color(dot) for dot in dots)

        self.add(dots)
        self.add(tail)

        self.play(
            *(ShowCreation(curve, rate_func=linear) for curve in curves),
            run_time=total_time,
        )
