#!/usr/bin/env python3
import os
import io
import json
import subprocess
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
VH_SCRIPT = ROOT / 'scripts' / 'visual_hull.py'
SAMPLE_PC_SCRIPT = ROOT / 'scripts' / 'mesh_to_pointcloud.py'


def run(cmd, cwd=None):
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    for line in proc.stdout:
        out_lines.append(line)
    code = proc.wait()
    return code, ''.join(out_lines)


def main():
    st.set_page_config(page_title='PMNI-flavored 3D App', layout='wide')
    st.title('Photos → 3D (PMNI-flavored, textureless mode)')
    st.caption('CPU-first pipeline: Visual Hull + mesh-sampled point cloud. GPU/PMNI path optional.')

    with st.expander('1) Select dataset folder (DiLiGenT-MV style)', expanded=True):
        default_dir = ROOT / 'data' / 'diligent_mv_normals' / 'bear'
        data_dir = st.text_input('Data directory', str(default_dir))
        views = st.number_input('Views (first N)', min_value=2, max_value=100, value=8)
        res = st.selectbox('Visual hull resolution (voxel grid)', [48, 64, 80, 96, 128], index=1)
        erode = st.slider('Mask erosion (pixels)', min_value=0, max_value=5, value=1)

    run_col1, run_col2 = st.columns(2)
    with run_col1:
        st.subheader('2) Visual Hull (CPU)')
        outdir_vh = st.text_input('Output dir (mesh)', str(ROOT / 'exp' / 'visual_hull'))
        name_vh = st.text_input('Mesh name', 'demo_hull')
        run_vh = st.button('Run Visual Hull')
    with run_col2:
        st.subheader('3) Point Cloud from Mesh (CPU)')
        outdir_pc = st.text_input('Output dir (point cloud)', str(ROOT / 'exp' / 'quick_outputs'))
        n_pts = st.number_input('Sample points', min_value=50000, max_value=1000000, value=250000, step=50000)
        run_pc = st.button('Sample Point Cloud from Mesh')

    vh_ply = Path(outdir_vh) / f"{name_vh}_v{views}_r{res}.ply"
    vh_png = Path(outdir_vh) / f"{name_vh}_v{views}_r{res}.png"

    if run_vh:
        st.write('Running visual hull…')
        code, log = run([
            'python3', str(VH_SCRIPT),
            '--data_dir', data_dir,
            '--views', str(int(views)),
            '--res', str(int(res)),
            '--erode', str(int(erode)),
            '--outdir', outdir_vh,
            '--name', name_vh
        ], cwd=str(ROOT))
        st.code(log)
        if code == 0 and vh_png.exists():
            st.success('Visual hull complete.')
            st.image(str(vh_png), caption=vh_png.name, use_column_width=True)
            prov = {
                'algorithm': 'visual_hull',
                'data_dir': os.path.abspath(data_dir),
                'views': int(views), 'res': int(res), 'erode': int(erode),
                'artifact': os.path.abspath(str(vh_ply))
            }
            prov_json = json.dumps(prov, indent=2).encode('utf-8')
            st.download_button('Download provenance (JSON)', data=prov_json, file_name=f'{name_vh}_provenance.json')
            if vh_ply.exists():
                with open(vh_ply, 'rb') as f:
                    st.download_button('Download mesh (PLY)', data=f.read(), file_name=vh_ply.name)

    if run_pc:
        if not vh_ply.exists():
            st.error('Run Visual Hull first to produce the mesh.')
        else:
            pc_ply = Path(outdir_pc) / f"{name_vh}_pc_from_mesh.ply"
            pc_png = Path(outdir_pc) / f"{name_vh}_pc_from_mesh.png"
            st.write('Sampling points from mesh…')
            code, log = run([
                'python3', str(SAMPLE_PC_SCRIPT),
                '--mesh', str(vh_ply),
                '--out_ply', str(pc_ply),
                '--out_png', str(pc_png),
                '--n_points', str(int(n_pts))
            ], cwd=str(ROOT))
            st.code(log)
            if code == 0 and pc_png.exists():
                st.success('Point cloud sampling complete.')
                st.image(str(pc_png), caption=pc_png.name, use_column_width=True)
                with open(pc_ply, 'rb') as f:
                    st.download_button('Download point cloud (PLY)', data=f.read(), file_name=pc_ply.name)

    st.divider()
    st.subheader('Future: COLMAP + PMNI‑lite refinement')
    st.caption('We will add automatic camera estimation via COLMAP for arbitrary photo sets and a PMNI‑inspired CPU refinement step for textureless scenes.')


if __name__ == '__main__':
    main()
