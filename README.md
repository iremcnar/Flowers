# Pointillism Flower 🌸

A generative pointillist art experiment in Python that simulates the complete lifecycle of a blooming flower—sprouting from textured soil, opening in multi-layered petal arrangements, emitting glowing aura particles, and gradually wilting over time.

<img width="1159" height="734" alt="WhatsApp Image 2026-10-15 at 13 41 41" src="https://github.com/user-attachments/assets/8a9a267d-7c49-4ae4-8477-2cb4343f8e6f" />


## Features
- **Procedural Pointillism:** Renders soil, stem, leaves, and petals using tens of thousands of dynamic point particles ("splats") rather than solid fills.
- **Keyframe-Driven Lifecycle:** Smoothly interpolates through distinct biological phases: soil disturbance, stem elongation, leaf unfurling, full blossom, and gradual decay.
- **Bézier Curves & 3D Projections:** Uses parametric cubic Bézier curves for the stem and projects layered 3D petal geometries with depth sorting and wind sway.
- **Atmospheric FX & Aura:** Features orbital energy rings, pulsing stamens, soft halos, and fluttering petal particles that detach upon wilting.
- **Interactive GUI & Batch Exporters:** Interactive Tkinter window with keyboard controls (`Space`, `R`, `S`, `Esc`), plus headless flags to export PNG stage strips or an animated GIF.

## How to Run

```bash
# Run interactive desktop viewer
python flower.py
