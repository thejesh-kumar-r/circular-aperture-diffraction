# Circular Aperture Diffraction: Fresnel and Fraunhofer Regimes

An interactive computational study of diffraction of monochromatic light through a circular aperture, demonstrating the transition from Fresnel (near-field) diffraction to Fraunhofer (far-field) diffraction.

## Overview

This project simulates the diffraction pattern produced when coherent light passes through a uniformly illuminated circular aperture.

The application uses numerical integration to calculate Fresnel diffraction and compares the result with the analytical Fraunhofer Airy pattern. By varying the experimental parameters, users can investigate how diffraction patterns evolve with propagation distance and how the Fresnel solution approaches the Fraunhofer limit.

## Objectives

- Numerically simulate Fresnel diffraction through a circular aperture.
- Study Fraunhofer diffraction and the formation of the Airy pattern.
- Investigate the effect of propagation distance, aperture radius and wavelength.
- Compare Fresnel and Fraunhofer intensity distributions.
- Calculate the error between the two diffraction models.
- Visualize the diffraction patterns using interactive 2D and 3D plots.

## Theoretical Background

### 1. Fresnel Diffraction

The axisymmetric Fresnel diffraction field for a circular aperture is proportional to

\[
U(r,z)\propto \int_0^a \rho
\exp\left(\frac{ik\rho^2}{2z}\right)
J_0\left(\frac{kr\rho}{z}\right)d\rho
\]

where \(a\) is the aperture radius, \(z\) is the propagation distance, \(k=2\pi/\lambda\), and \(J_0\) is the zeroth-order Bessel function.

### 2. Fraunhofer Diffraction

In the far-field approximation, the intensity distribution follows the Airy pattern:

\[
I(r)=I(0)\left[\frac{2J_1(v)}{v}\right]^2
\]

where

\[
v=\frac{2\pi ar}{\lambda z}
\]

and \(J_1\) is the first-order Bessel function.

### 3. Fresnel Number

The Fresnel number characterizes the diffraction regime:

\[
N_F=\frac{a^2}{\lambda z}
\]

A large Fresnel number corresponds to near-field diffraction, while a small Fresnel number indicates far-field behaviour.

## Interactive Features

- Adjustable screen distance.
- Adjustable circular aperture radius.
- Adjustable wavelength.
- Interactive Fresnel diffraction intensity maps.
- Fraunhofer Airy diffraction visualization.
- Radial intensity distribution graphs.
- Three-dimensional intensity visualization.
- Fresnel versus Fraunhofer comparison.
- RMS error analysis and convergence with distance.

## Technologies Used

- Python
- NumPy
- SciPy
- Streamlit
- Plotly
- Matplotlib

## Installation and Execution

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/circular-aperture-diffraction.git
```

Navigate to the project directory:

```bash
cd circular-aperture-diffraction
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The interactive application will open in your web browser.

## Project Structure

```text
circular-aperture-diffraction/
│
├── app.py
├── requirements.txt
└── README.md
```

## Model Assumptions

- Monochromatic and coherent illumination.
- Uniform illumination across the circular aperture.
- Scalar and paraxial diffraction approximation.
- Thin aperture in an opaque screen.
- Rotational symmetry about the optical axis.

## Expected Outcome

The simulation demonstrates the evolution of the Fresnel diffraction pattern with propagation distance and its convergence towards the Fraunhofer Airy pattern in the far-field regime.

## Author

Engineering Physics Undergraduate Project

## Acknowledgements

Developed as part of an optics and diffraction computational study.
