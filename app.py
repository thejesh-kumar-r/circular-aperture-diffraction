import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import j0, j1
from numpy.polynomial.legendre import leggauss

# ------------------------------------------------------------
# Circular-aperture Fresnel / Fraunhofer diffraction simulator
# ------------------------------------------------------------

st.set_page_config(
    page_title="Circular Aperture Diffraction",
    page_icon="🔬",
    layout="wide",
)

st.title("Single Circular Aperture: Fresnel → Fraunhofer Diffraction")
st.caption(
    "One numerical model for near-field Fresnel propagation, "
    "far-field Airy diffraction, and their convergence with distance."
)

st.markdown(
    """
    The aperture is a **clear circular hole of radius** \(a\) in an opaque screen.
    The calculation uses the scalar, paraxial Fresnel diffraction integral.
    The Fraunhofer Airy pattern is calculated independently as the far-field reference.
    """
)

# -------------------------- Sidebar --------------------------

st.sidebar.header("Experimental parameters")

z_m = st.sidebar.slider(
    "Aperture-to-screen distance, z (m)",
    min_value=0.05, max_value=20.0, value=2.0, step=0.05,
)
a_mm = st.sidebar.slider(
    "Circular aperture radius, a (mm)",
    min_value=0.05, max_value=1.50, value=0.25, step=0.01,
)
wavelength_nm = st.sidebar.slider(
    "Wavelength, λ (nm)",
    min_value=380, max_value=750, value=550, step=5,
)
n_radial = st.sidebar.select_slider(
    "Radial calculation resolution",
    options=[250, 350, 450, 550, 700, 850],
    value=450,
)
n_quad = st.sidebar.select_slider(
    "Fresnel integration resolution",
    options=[120, 180, 240, 320, 400],
    value=240,
)
map_points = st.sidebar.select_slider(
    "2D map resolution",
    options=[151, 201, 251, 301, 401],
    value=301,
)

display_half_width_mm = st.sidebar.select_slider(
    "Screen display half-width (mm)",
    options=[
        0.02, 0.05, 0.1, 0.2, 0.5,
        1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0
    ],
    value=20.0,
)

a = a_mm * 1e-3
lam = wavelength_nm * 1e-9
z = z_m
k = 2.0 * np.pi / lam
fresnel_number = a * a / (lam * z)

# Use the dimensionless radial coordinate u = k*a*r/z.
# The Fraunhofer Airy profile is [2 J1(u) / u]^2.
u_max = 22.0
u = np.linspace(0.0, u_max, n_radial)
r = u * z / (k * a)  # physical screen radius, metres
r_mm = r * 1e3


@st.cache_data(show_spinner=False)
def gauss_legendre_nodes(n):
    """Gauss-Legendre nodes and weights mapped from [-1,1] to [0,1]."""
    x, w = leggauss(n)
    return (x + 1.0) / 2.0, w / 2.0


@st.cache_data(show_spinner="Calculating Fresnel diffraction...")
def fresnel_radial_intensity(z_value, a_value, lam_value, n_r, n_q, u_limit):
    """
    Evaluate the axisymmetric Fresnel diffraction integral.

    U(r,z) is proportional to:
      integral_0^a rho * exp(i*k*rho^2/(2z))
                    * J0(k*r*rho/z) d(rho)

    The omitted propagation prefactor is independent of screen radius, so
    normalizing the intensity profile removes it without changing the pattern.
    """
    kval = 2.0 * np.pi / lam_value
    uu = np.linspace(0.0, u_limit, n_r)
    rr = uu * z_value / (kval * a_value)

    # Transform rho = a*t, t in [0,1].
    t, weights = gauss_legendre_nodes(n_q)
    rho = a_value * t
    integration_weights = a_value * weights

    phase = np.exp(1j * kval * rho**2 / (2.0 * z_value))
    bessel_argument = (kval * rr[:, None] * rho[None, :]) / z_value

    integrand = (
        j0(bessel_argument)
        * phase[None, :]
        * rho[None, :]
    )
    field = integrand @ integration_weights
    intensity = np.abs(field) ** 2

    peak = np.max(intensity)
    if peak > 0:
        intensity = intensity / peak
    return uu, rr, intensity


def airy_intensity(u_values):
    """Normalized Fraunhofer Airy intensity for a uniformly illuminated disk."""
    out = np.ones_like(u_values, dtype=float)
    nonzero = np.abs(u_values) > 1e-12
    out[nonzero] = (
        2.0 * j1(u_values[nonzero]) / u_values[nonzero]
    ) ** 2
    return out


u_calc, r_calc, fresnel_I = fresnel_radial_intensity(
    float(z), float(a), float(lam), int(n_radial), int(n_quad), float(u_max)
)
airy_I = airy_intensity(u_calc)

# Both profiles are normalized to their own peak for a shape comparison.
# This deliberately compares pattern shape, not absolute irradiance.
difference = fresnel_I - airy_I
rms_error = float(np.sqrt(np.mean(difference**2)))
max_abs_error = float(np.max(np.abs(difference)))

# -------------------------- Summary metrics --------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Fresnel number", f"{fresnel_number:.4g}")
col2.metric("Normalized RMS shape error", f"{rms_error:.5f}")
col3.metric("Maximum absolute shape error", f"{max_abs_error:.5f}")
col4.metric("First Airy minimum", f"{1.22 * lam * z / (2*a) * 1e3:.3f} mm")

if fresnel_number > 1:
    regime_text = "Fresnel / near-field dominated"
elif fresnel_number > 0.1:
    regime_text = "Transition region"
else:
    regime_text = "Fraunhofer-like / far-field"

st.info(
    f"**Current regime estimate:** {regime_text}. "
    "The Fresnel number shown here uses the aperture radius: "
    r" \(N_F=a^2/(\lambda z)\). "
    "The boundaries are guides, not sharp physical cutoffs."
)

# -------------------------- Main profile graph --------------------------

st.subheader("1. Radial intensity: Fresnel calculation vs Fraunhofer reference")

profile_mask = r_mm <= display_half_width_mm

r_display = r_mm[profile_mask]
fresnel_display = fresnel_I[profile_mask]
airy_display = airy_I[profile_mask]

profile_fig = go.Figure()
profile_fig.add_trace(go.Scatter(
    x=r_display,
    y=fresnel_display,
    mode="lines",
    name="Fresnel numerical result",
    line=dict(width=2.5),
))

profile_fig.add_trace(go.Scatter(
    x=r_display,
    y=airy_display,
    mode="lines",
    name="Fraunhofer Airy reference",
    line=dict(width=2, dash="dash"),
))
profile_fig.update_layout(
    xaxis_title="Radial screen coordinate r (mm)",
    yaxis_title="Normalized intensity",
    yaxis=dict(range=[0, 1.05]),
    legend=dict(orientation="h", y=1.12),
    height=440,
    margin=dict(l=20, r=20, t=55, b=20),
)
st.plotly_chart(profile_fig, use_container_width=True)

st.caption(
    "Both curves are independently normalized to their own on-axis peak. "
    "This makes the comparison focus on the shape of the diffraction pattern."
)

# -------------------------- 2D intensity map --------------------------

st.subheader("2. Two-dimensional Fresnel diffraction pattern")

# Fixed physical screen window, independent of z, a and wavelength
r_max = display_half_width_mm * 1e-3

axis = np.linspace(-r_max, r_max, map_points)
XX, YY = np.meshgrid(axis, axis)

RR = np.sqrt(XX**2 + YY**2)

radial_u_map = (k * a * RR) / z

# Interpolate numerical Fresnel intensity
I_map = np.interp(
    radial_u_map.ravel(),
    u_calc,
    fresnel_I,
    left=fresnel_I[0],
    right=fresnel_I[-1]
).reshape(RR.shape)

# Logarithmic intensity for revealing weak secondary rings
I_map_db = 10 * np.log10(np.maximum(I_map, 1e-6))

map_fig = go.Figure(data=go.Heatmap(
    x=axis * 1e3,
    y=axis * 1e3,
    z=I_map_db,
    colorscale="Inferno",
    zmin=-60,
    zmax=0,
    colorbar=dict(
        title="Intensity (dB)",
        ticksuffix=" dB"
    ),
))

map_fig.update_layout(
    xaxis_title="x (mm)",
    yaxis_title="y (mm)",
    yaxis_scaleanchor="x",
    height=650,
    margin=dict(l=20, r=20, t=20, b=20),
)

st.plotly_chart(map_fig, use_container_width=True)

# -------------------------- 3D surface --------------------------

with st.expander("3. 3D intensity surface", expanded=False):
    surface_fig = go.Figure(data=[go.Surface(
        x=XX * 1e3,
        y=YY * 1e3,
        z=I_map,
        colorscale="Inferno",
        colorbar=dict(title="I / Imax"),
    )])
    surface_fig.update_layout(
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="Normalized intensity",
            aspectratio=dict(x=1, y=1, z=0.55),
        ),
        height=650,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(surface_fig, use_container_width=True)

# -------------------------- Error vs distance --------------------------

st.subheader("4. Convergence: error as the screen moves farther away")

st.markdown(
    """
    This graph recomputes the Fresnel integral at a sequence of screen distances.
    At each distance, the Fresnel profile is compared with the normalized Airy
    profile on the same dimensionless radial interval.
    """
)

z_min_error = max(0.05, min(z_m, 0.25))
z_max_error = max(20.0, z_m)
number_of_distances = 24
distance_values = np.geomspace(z_min_error, z_max_error, number_of_distances)

error_values = []
nf_values = []

progress = st.progress(0, text="Calculating distance sweep...")
for idx, z_test in enumerate(distance_values):
    u_test, _, I_test = fresnel_radial_intensity(
        float(z_test), float(a), float(lam),
        min(int(n_radial), 450), min(int(n_quad), 240), float(u_max)
    )
    I_airy_test = airy_intensity(u_test)
    error_values.append(float(np.sqrt(np.mean((I_test - I_airy_test) ** 2))))
    nf_values.append(a * a / (lam * z_test))
    progress.progress((idx + 1) / number_of_distances,
                       text=f"Calculating distance sweep... {idx + 1}/{number_of_distances}")
progress.empty()

error_fig = make_subplots(specs=[[{"secondary_y": True}]])
error_fig.add_trace(go.Scatter(
    x=distance_values, y=error_values,
    mode="lines+markers", name="Normalized RMS error",
), secondary_y=False)
error_fig.add_trace(go.Scatter(
    x=distance_values, y=nf_values,
    mode="lines", name="Fresnel number",
    line=dict(dash="dash"),
), secondary_y=True)
error_fig.update_xaxes(
    title_text="Screen distance z (m)",
    type="log",
    tickmode="array",
    tickvals=[
        0.25, 0.3, 0.4, 0.5, 0.6, 0.8,
        1, 2, 3, 4, 5, 6, 8, 10, 20
    ],
    ticktext=[
        "0.25", "0.3", "0.4", "0.5", "0.6", "0.8",
        "1", "2", "3", "4", "5", "6", "8", "10", "20"
    ],
)
error_fig.update_yaxes(
    title_text="Normalized RMS shape error",
    rangemode="tozero",
    secondary_y=False,
)
error_fig.update_yaxes(
    title_text="Fresnel number",
    type="log",
    tickmode="array",
    tickvals=[0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 3],
    ticktext=[
        "0.03", "0.05", "0.1", "0.2",
        "0.5", "1", "2", "3"
    ],
    secondary_y=True,
)
error_fig.update_layout(
    height=480,
    legend=dict(orientation="h", y=1.12),
    margin=dict(l=20, r=20, t=55, b=20),
)
st.plotly_chart(error_fig, use_container_width=True)

# -------------------------- Physics and numerical notes --------------------------

with st.expander("5. Equations, assumptions and interpretation"):
    st.latex(
        r"U(r,z)\\propto \\int_0^a \\rho\\,"
        r"e^{i k\\rho^2/(2z)}"
        r"J_0\\!\\left(\\frac{k r\\rho}{z}\\right)d\\rho"
    )
    st.latex(
        r"I_{\\mathrm{Airy}}(r)=I(0)"
        r"\\left[\\frac{2J_1(v)}{v}\\right]^2,"
        r"\\qquad v=\\frac{k a r}{z}"
    )
    st.latex(r"N_F=\\frac{a^2}{\\lambda z}")
    st.markdown(
        """
        **Model assumptions**
        - Monochromatic, coherent, uniformly illuminated circular aperture.
        - Scalar diffraction and the paraxial Fresnel approximation.
        - The aperture is thin and lies in an otherwise opaque screen.
        - The calculation assumes rotational symmetry about the optical axis.
        - The propagation prefactor is omitted in the displayed Fresnel field
          because the plots are normalized to peak intensity.

        **Important interpretation**
        The Fresnel integral is a paraxial approximation to propagation, not an
        exact non-paraxial electromagnetic solution. The far-field Airy pattern
        is its Fraunhofer limit. A smaller normalized RMS error indicates closer
        agreement in pattern shape over the displayed radial interval.
        """
    )

st.subheader("6. Numerical settings")
st.write(
    f"Radial samples: {n_radial} · "
    f"Gauss-Legendre integration points: {n_quad} · "
    f"2D map: {map_points} × {map_points}"
)
st.caption(
    "For final report-quality results, increase the radial and integration "
    "resolutions and verify that the plotted curves and error values converge."
)
