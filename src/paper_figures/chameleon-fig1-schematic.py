import graphviz

def generate_chameleon_figure():
    # Initialize the directed graph with standard NeurIPS-friendly fonts
    dot = graphviz.Digraph('Chameleon_Framework', format='png')
    # ordering='out' pins each node's out-edges to declaration order, which in
    # rankdir=LR fixes the top-to-bottom order of the fork column to the order
    # the paper introduces the forks (UNet -> few-step -> DiT).
    dot.attr(rankdir='LR', ordering='out', fontname='Helvetica', fontsize='12')

    # Define common node styles
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#f8f9fa',
             fontname='Helvetica', fontsize='18', color='#343a40', penwidth='1.5')
    dot.attr('edge', fontname='Helvetica', color='#495057', penwidth='1.2', arrowhead='vee')

    # ── Offline analysis: TWO parallel paths ──────────────────────────────────
    # (1) Weights — quantized once, statically, per layer.
    # (2) Activations — routed per layer AND per timestep bucket at run time.
    with dot.subgraph(name='cluster_offline') as c:
        c.attr(label='Offline Analysis', style='dashed', color='#adb5bd', bgcolor='#ffffff',
               fontname='Helvetica-Bold', fontsize='20')

        # Weight path (static, timestep-independent)
        c.node('WStats', 'Weight SNR\n(per output-channel)', fillcolor='#e9ecef')
        c.node('WPalette',
               'Adaptive Weight Palette\n(group, format) search\nW8: {INT8, MXINT8}\nW4: {INT4, NF4, FP4, MX4}',
               fillcolor='#e8f5e9')
        c.edge('WStats', 'WPalette')

        # Activation path (timestep-aware) — Timestep t feeds the diffusion-SNR routing
        c.node('Tstep', 'Timestep t', fillcolor='#fffde7')
        c.node('AStats', 'Activation Statistics\n(Kurtosis κ, Diffusion SNR(t))', fillcolor='#e9ecef')
        c.node('ALUT',
               'Per-(Layer × Timestep-Bucket)\nActivation Format LUT\n{INT8, FP8, MX}',
               fillcolor='#e3f2fd')
        c.edge('Tstep', 'AStats')
        c.edge('AStats', 'ALUT')

        # Align into a clean two-row layout: weights (upper) and activations
        # (lower) share ranks so the two parallel paths read left-to-right.
        c.body.append('{rank=same; WStats; AStats}')
        c.body.append('{rank=same; WPalette; ALUT}')

    # Subgraph for the Architectural Fork
    with dot.subgraph(name='cluster_fork') as c:
        c.attr(label='Load-Time Architectural Fork', style='dashed', color='#adb5bd', bgcolor='#ffffff',
               fontname='Helvetica-Bold', fontsize='20')
        c.node('UNet', 'U-Net Diffusion\n(Q-Diffusion style)', fillcolor='#fce4ec')
        c.node('LCM', 'Few-Step Distilled\n(SDXL-Turbo, MixDQ style)', fillcolor='#fff3e0')
        c.node('DiT', 'Diffusion Transformer\n(Q-DiT style)', fillcolor='#f3e5f5')

        # Pin the vertical order to the order the paper introduces the forks
        # (UNet -> few-step -> DiT). Without this graphviz reorders the rank to
        # minimise edge crossings from WPalette/ALUT.
        c.body.append('{rank=same; UNet; LCM; DiT}')
        # NB: rankdir=LR rotates the layout, so a flat edge A -> B places A
        # BELOW B. The edges are therefore written in reverse of the desired
        # top-to-bottom order.
        c.body.append('DiT -> LCM [style=invis]')
        c.body.append('LCM -> UNet [style=invis]')

    # Subgraph for Inference Execution — now naming BOTH weights and activations,
    # and the timestep regime each family operates in.
    with dot.subgraph(name='cluster_inference') as c:
        c.attr(label='Inference Execution Graphs', style='dashed', color='#adb5bd', bgcolor='#ffffff',
               fontname='Helvetica-Bold', fontsize='20')
        c.node('Exec_UNet',
               'Fake-Quant Dispatch\nW: per-channel (g,f)\nA: per-bucket dynamic + MXFP8 shortcuts')
        c.node('Exec_LCM',
               'Fake-Quant Dispatch\nW: adaptive palette\nA: MixDQ static scales (1-step)')
        c.node('Exec_DiT',
               'Fake-Quant Dispatch\nW: input-aware (g,f) selection\nA: macro-route + per-tensor micro-scaling')

        # Same pinning so each execution graph sits beside its fork.
        c.body.append('{rank=same; Exec_UNet; Exec_LCM; Exec_DiT}')
        c.body.append('Exec_DiT -> Exec_LCM [style=invis]')
        c.body.append('Exec_LCM -> Exec_UNet [style=invis]')

    # Both the weight palette and the activation LUT are consumed by every
    # architecture; the load-time fork then specialises the execution graph.
    # The weight palette feeds every architecture. The activation LUT feeds only
    # the multi-step paths: the 1-step few-step model has no timestep axis to
    # route over, so its activations come from MixDQ's calibrated static scales.
    for arch in ('UNet', 'LCM', 'DiT'):
        dot.edge('WPalette', arch)
    for arch in ('UNet', 'DiT'):
        dot.edge('ALUT', arch)
    # Layout-only: LCM has one incoming edge where UNet/DiT have two, which makes
    # dot's mincross pull it to the top of the fork column. This invisible edge
    # restores the symmetry so the column keeps the declared UNet -> LCM -> DiT
    # order. It carries no meaning: the few-step path consumes no activation LUT.
    dot.edge('ALUT', 'LCM', style='invis')

    dot.edge('UNet', 'Exec_UNet')
    dot.edge('LCM', 'Exec_LCM')
    dot.edge('DiT', 'Exec_DiT')

    # Render to file
    dot.render('chameleon_schematic', cleanup=True)
    print("Figure generated successfully as chameleon_schematic.png")

if __name__ == "__main__":
    generate_chameleon_figure()
