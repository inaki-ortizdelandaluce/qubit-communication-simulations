import numpy as np
import matplotlib
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D


class Arrow3D(Patch3D, FancyArrowPatch):
    """Makes a fancy arrow"""

    # Nasty hack around a poorly implemented deprecation warning in Matplotlib 3.5 that issues two
    # deprecation warnings if an artist's module does not claim to be part of the below module.
    # This revolves around the method `Patch3D.do_3d_projection(self, renderer=None)`.  The
    # `renderer` argument has been deprecated since Matplotlib 3.4, but in 3.5 some internal calls
    # during `Axes3D` display started calling the method.  If an artist does not have this module,
    # then it issues a deprecation warning, and calls it by passing the `renderer` parameter as
    # well, which consequently triggers another deprecation warning.  We should be able to remove
    # this once 3.6 is the minimum supported version, because the deprecation period ends then.
    __module__ = "mpl_toolkits.mplot3d.art3d"

    def __init__(self, xs, ys, zs, zdir="z", **kwargs):
        # The Patch3D.__init__() method just calls its own super() method and then
        # self.set_3d_properties, but its __init__ signature is actually pretty incompatible with
        # how it goes on to call set_3d_properties, so we just have to do things ourselves.  The
        # parent of Patch3D is Patch, which is also a parent of FancyArrowPatch, so its __init__ is
        # still getting suitably called.
        # pylint: disable=super-init-not-called
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), **kwargs)
        self.set_3d_properties(tuple(zip(xs, ys)), zs, zdir)
        self._path2d = None

    def draw(self, renderer):
        xs3d, ys3d, zs3d = zip(*self._segment3d)
        x_s, y_s, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self._path2d = matplotlib.path.Path(np.column_stack([x_s, y_s]))
        self.set_positions((x_s[0], y_s[0]), (x_s[1], y_s[1]))
        FancyArrowPatch.draw(self, renderer)


def add_inset_axes(rect, units="ax", ax_target=None, fig=None, projection=None, **kw):
    from matplotlib.transforms import Bbox
    """
    Wrapper around `fig.add_axes` to achieve `ax.inset_axes` functionality
    that works also for insetting 3D plot on 2D ax/figures
    """
    assert ax_target is not None or fig is not None, "`fig` or `ax_target` must be provided!"
    _units = {"ax", "norm2ax", "norm2fig"}
    assert {units} <= _units, "`rect_units` not in {}".format(repr(_units))

    if ax_target is not None:
        # Inspired from:
        # https://stackoverflow.com/questions/14568545/convert-matplotlib-data-units-to-normalized-units
        bb_data = Bbox.from_bounds(*rect)
        trans = ax_target.transData if units == "ax" else ax_target.transAxes
        disp_coords = trans.transform(bb_data)
        fig = ax_target.get_figure()
        fig_coord = fig.transFigure.inverted().transform(disp_coords)
    elif fig is not None:
        if ax_target is not None and units != "norm2fig":
            bb_data = Bbox.from_bounds(*rect)
            trans = ax_target.transData if units == "ax" else ax_target.transAxes
            disp_coords = trans.transform(bb_data)
        else:
            fig_coord = Bbox.from_bounds(*rect)

    axin = fig.add_axes(
        Bbox(fig_coord),
        projection=projection, **kw)

    return axin


def plot_sphere_back(axes, frame_alpha=0.2, frame_color="gray", frame_width=1, sphere_color="#FFDDDD",
                     sphere_alpha=0.2):
    """back half of sphere"""
    u_angle = np.linspace(0, np.pi, 25)
    v_angle = np.linspace(0, np.pi, 25)
    x_dir = np.outer(np.cos(u_angle), np.sin(v_angle))
    y_dir = np.outer(np.sin(u_angle), np.sin(v_angle))
    z_dir = np.outer(np.ones(u_angle.shape[0]), np.cos(v_angle))
    axes.plot_surface(
        x_dir,
        y_dir,
        z_dir,
        rstride=2,
        cstride=2,
        color=sphere_color,
        linewidth=0,
        alpha=sphere_alpha,
    )
    # wireframe
    axes.plot_wireframe(
        x_dir,
        y_dir,
        z_dir,
        rstride=5,
        cstride=5,
        color=frame_color,
        alpha=frame_alpha,
    )
    # equator
    axes.plot(
        1.0 * np.cos(u_angle),
        1.0 * np.sin(u_angle),
        zs=0,
        zdir="z",
        lw=frame_width,
        color=frame_color,
    )
    axes.plot(
        1.0 * np.cos(u_angle),
        1.0 * np.sin(u_angle),
        zs=0,
        zdir="x",
        lw=frame_width,
        color=frame_color,
    )


def plot_sphere_front(axes, frame_alpha=0.2, frame_color="gray", frame_width=1, sphere_color="#FFDDDD",
                      sphere_alpha=0.2):
    u_angle = np.linspace(-np.pi, 0, 25)
    v_angle = np.linspace(0, np.pi, 25)
    x_dir = np.outer(np.cos(u_angle), np.sin(v_angle))
    y_dir = np.outer(np.sin(u_angle), np.sin(v_angle))
    z_dir = np.outer(np.ones(u_angle.shape[0]), np.cos(v_angle))
    axes.plot_surface(
        x_dir,
        y_dir,
        z_dir,
        rstride=2,
        cstride=2,
        color=sphere_color,
        linewidth=0,
        alpha=sphere_alpha,
    )
    # wireframe
    axes.plot_wireframe(
        x_dir,
        y_dir,
        z_dir,
        rstride=5,
        cstride=5,
        color=frame_color,
        alpha=frame_alpha,
    )
    # equator
    axes.plot(
        1.0 * np.cos(u_angle),
        1.0 * np.sin(u_angle),
        zs=0,
        zdir="z",
        lw=frame_width,
        color=frame_color,
    )
    axes.plot(
        1.0 * np.cos(u_angle),
        1.0 * np.sin(u_angle),
        zs=0,
        zdir="x",
        lw=frame_width,
        color=frame_color,
    )
    return None


def plot_vectors(axes, vectors, vector_style="-|>", vector_mutation=20,
                 vector_color=["#dc267f", "#648fff", "#fe6100", "#785ef0", "#ffb000"], vector_width=3):
    # -X and Y data are switched for plotting purposes
    for k in range(len(vectors)):

        xs3d = vectors[k][1] * np.array([0, 1])
        ys3d = -vectors[k][0] * np.array([0, 1])
        zs3d = vectors[k][2] * np.array([0, 1])

        color = vector_color[np.mod(k, len(vector_color))]

        if vector_style == "":
            # simple line style
            axes.plot(
                xs3d, ys3d, zs3d, zs=0, zdir="z", label="Z", lw=vector_width, color=color
            )
        else:
            # decorated style, with arrow heads
            arr = Arrow3D(
                xs3d,
                ys3d,
                zs3d,
                mutation_scale=vector_mutation,
                lw=vector_width,
                arrowstyle=vector_style,
                color=color,
            )

            axes.add_artist(arr)
    return None


def plot_axes_labels(axes,
                     font_size=20, font_color="gray",
                     xlabel=["", ""], xlpos=[1.2, -1.2],
                     ylabel=["", ""], ylpos=[1.2, -1.2],
                     zlabel=[r"$\left|0\right>$", r"$\left|1\right>$"], zlpos=[1.2, -1.2]):
    opts = {
        "fontsize": font_size,
        "color": font_color,
        "horizontalalignment": "center",
        "verticalalignment": "center",
    }
    axes.text(0, -xlpos[0], 0, xlabel[0], **opts)
    axes.text(0, -xlpos[1], 0, xlabel[1], **opts)

    axes.text(ylpos[0], 0, 0, ylabel[0], **opts)
    axes.text(ylpos[1], 0, 0, ylabel[1], **opts)

    axes.text(0, 0, zlpos[0], zlabel[0], **opts)
    axes.text(0, 0, zlpos[1], zlabel[1], **opts)

    for item in axes.xaxis.get_ticklines() + axes.xaxis.get_ticklabels():
        item.set_visible(False)
    for item in axes.yaxis.get_ticklines() + axes.yaxis.get_ticklabels():
        item.set_visible(False)
    for item in axes.zaxis.get_ticklines() + axes.zaxis.get_ticklabels():
        item.set_visible(False)
    return None


def plot_annotations(axes, annotations, font_size=20, font_color="#000000"):
    # -X and Y data are switched for plotting purposes
    for annotation in annotations:
        vec = annotation["position"]
        opts = {
            "fontsize": font_size,
            "color": font_color,
            "horizontalalignment": "center",
            "verticalalignment": "center",
        }
        opts.update(annotation["opts"])
        axes.text(vec[1], -vec[0], vec[2], annotation["text"], **opts)
    return None


def plot_bloch_sphere(axes, state, measurements,
                        vector_color=["#dc267f", "#648fff", "#fe6100", "#785ef0", "#ffb000"]):
    frame_width = 1
    frame_color = "gray"
    axes.clear()
    # plot axes
    span = np.linspace(-1.0, 1.0, 2)
    axes.plot(
        span, 0 * span, zs=0, zdir="z", label="X", lw=frame_width, color=frame_color
    )
    axes.plot(
        0 * span, span, zs=0, zdir="z", label="Y", lw=frame_width, color=frame_color
    )
    axes.plot(
        0 * span, span, zs=0, zdir="y", label="Z", lw=frame_width, color=frame_color
    )
    axes.set_axis_off()
    axes.set_xlim3d(-0.7, 0.7)
    axes.set_ylim3d(-0.7, 0.7)
    axes.set_zlim3d(-0.7, 0.7)
    axes.set_box_aspect((1, 1, 1))
    axes.grid(False)
    plot_sphere_back(axes)
    plot_vectors(axes, measurements, vector_color=vector_color, vector_width=2)
    plot_vectors(axes, state, vector_color=["#000000"], vector_width=2)
    plot_sphere_front(axes)
    plot_axes_labels(axes, font_size=10)
    annotations = []
    state_name = "q"  # "$\\Psi$"
    annotations.append({"position": state[0], "text": state_name, "opts": {"fontsize": 10, "color": "#000000",
                                                                      "horizontalalignment": "center",
                                                                      "verticalalignment": "center"}})
    for idx, m in enumerate(measurements):
        name = r"$E_{}$".format(idx+1)
        annotations.append({"position": m, "text": name, "opts": {"fontsize": 12, "color": "#000000",
                                                                  "horizontalalignment": "center",
                                                                  "verticalalignment": "center"}})
    plot_annotations(axes, annotations)
    axes.set_title('', fontsize=20, y=1.08)