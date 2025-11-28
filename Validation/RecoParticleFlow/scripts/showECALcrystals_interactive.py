"""
To get bokeh on lxplus9:
source /cvmfs/sft.cern.ch/lcg/releases/LCG_108/Python/3.12.11/x86_64-el9-gcc13-opt/Python-env.sh

To get missing software (libarrow as an example):
source /cvmfs/sft.cern.ch/lcg/releases/LCG_108/pyarrow/20.0.0/x86_64-el9-gcc13-opt/pyarrow-env.sh
"""

import os
import argparse
import uproot
import awkward as ak
import utils
import numpy as np
import pandas as pd
from dataclasses import dataclass

from bokeh.plotting import figure, output_file, save, ColumnDataSource
from bokeh.models import HoverTool, Rect, ColumnDataSource, LogColorMapper, ColorBar, NumericInput, Dropdown, CDSView, GroupFilter, BooleanFilter, CustomJS, Slider
from bokeh.palettes import Viridis256, Category10
from bokeh.transform import linear_cmap, log_cmap
from bokeh.layouts import layout

def createFigure(title):
    fig = figure(
        title=title,
        x_axis_label=r"$$\eta$$",
        y_axis_label=r"$$\phi$$",
        width=1100,
        height=700,
        tools="pan,wheel_zoom,box_zoom,undo,redo,reset,save",
        active_drag="box_zoom",
        active_scroll=None
    )
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    fig.toolbar.logo = None
    return fig

def plotGeom(df, output_path):
    """Plot ECAL geometry as interactive Bokeh plot."""
    output_file(output_path)

    df['width'] = utils.angleDiff(df.crystalCorner2Eta, df.crystalCorner0Eta)
    df['height'] = utils.angleDiff(df.crystalCorner2Phi, df.crystalCorner0Phi)
    # df = df.iloc[0:20000]
    source = ColumnDataSource(df)

    # Create figure
    p = createFigure(title='ECAL-Barrel Geometry')
    
    # Add rectangles for each crystal    
    p.add_tools(HoverTool(
        tooltips=[
            ("DetID", "@crystalDetId"),
            ("Center (η, φ)", "(@crystalCenterEta, @crystalCenterPhi)"),
        ],
        mode="mouse",
    ))

    # Use rect glyph for each crystal
    p.rect(
        x="crystalCenterEta",
        y="crystalCenterPhi",
        width="width",
        height="height",
        source=source,
        fill_color="lightgray",
        line_color="black",
        alpha=0.1,
    )

    # Add scatter for centers
    p.scatter(
        x="crystalCenterEta",
        y="crystalCenterPhi",
        source=source,
        color="red",
        size=6,
    )
    p.scatter(
        x="crystalCorner2Eta",
        y="crystalCorner2Phi",
        source=source,
        color="green",
        size=6,
    )
    p.scatter(
        x="crystalCorner0Eta",
        y="crystalCorner0Phi",
        source=source,
        color="blue",
        size=6,
    )

    save(p)
    print(f"INFO: Geometry plot saved to {output_path}")

def shift_phi_corners(phi0, phi1, phi2, phi3):
    corners = [phi0, phi1, phi2, phi3]
    # Check each pair of adjacent corners
    for i in range(4):
        j = (i + 1) % 4
        diff = abs(corners[i] - corners[j])
        if diff > np.pi:
            # Shift the larger value by -2π
            if corners[i] > corners[j]:
                corners[i] -= 2 * np.pi
            else:
                corners[j] -= 2 * np.pi
    return corners + [corners[0]]  # Close the patch

def plotEvent(geom, hits, clusters, hits_in_clusters, output_path,
              variables=("energy", "frac"), zlabel=""):
    """Plot single event on top of the geometry, with interactive hover."""
    output_file(output_path)

    modes = ('Sim', 'Reco')

    p, src, srcCluster, df, hover, view, threshFilter = ({} for _ in range(7))
    for mode in modes:
        df[mode] = pd.merge(hits_in_clusters[mode], geom, how="inner", left_on="detids", right_on="crystalDetId")
        df[mode] = df[mode][df[mode].eventId < 100]
        df[mode]["eventId"] = df[mode]["eventId"].astype(str)


        clusters[mode]["eventId"] = clusters[mode]["eventId"].astype(str)
        srcCluster[mode] = ColumnDataSource(clusters[mode])

        view[mode] = CDSView(filters=[
            GroupFilter(column_name="eventId", group="1"),
            BooleanFilter([True] * len(df[mode]))]
        )
        threshFilter[mode] = view[mode].filters[1]
        
        # Create lists of lists for xs and ys
        df[mode]['xs'] = [
            [eta1, eta2, eta3, eta4, eta1]  # Close the patch by repeating the first point
            for eta1, eta2, eta3, eta4 in zip(
                    df[mode]["crystalCorner0Eta"], df[mode]["crystalCorner1Eta"],
                    df[mode]["crystalCorner2Eta"], df[mode]["crystalCorner3Eta"]
            )
        ]
        df[mode]['ys'] = [
            shift_phi_corners(phi0, phi1, phi2, phi3)  # Close the patch by repeating the first point
            for phi0, phi1, phi2, phi3 in zip(
                    df[mode]["crystalCorner0Phi"], df[mode]["crystalCorner1Phi"],
                    df[mode]["crystalCorner2Phi"], df[mode]["crystalCorner3Phi"]
            )
        ]

        # Aggregate data for unique patches
        patch_data = pd.DataFrame({
            'eventId': df[mode]['eventId'],
            'xs': df[mode]['xs'],
            'ys': df[mode]['ys'],
            'energies': df[mode]['energies'],
            'fracs': df[mode]['fracs'],
        })

        patch_data['xs_tuple'] = patch_data['xs'].apply(tuple)
        patch_data['ys_tuple'] = patch_data['ys'].apply(tuple)
        aggregated = patch_data.groupby(['xs_tuple', 'ys_tuple'], as_index=False).agg({
            'energies': 'sum',
            'fracs': 'sum',
        })

        aggregated['xs'] = aggregated['xs_tuple'].apply(list)
        aggregated['ys'] = aggregated['ys_tuple'].apply(list)

        energy_sum_map = aggregated.set_index(['xs_tuple', 'ys_tuple'])['energies'].to_dict()
        frac_sum_map = aggregated.set_index(['xs_tuple', 'ys_tuple'])['fracs'].to_dict()

        # Add 'energy_sum' to the original DataFrame
        df[mode]['energies_sum'] = df[mode].apply(
            lambda row: energy_sum_map.get((tuple(row['xs']), tuple(row['ys'])), 0),
            axis=1
        )
        df[mode]['fracs_sum'] = df[mode].apply(
            lambda row: frac_sum_map.get((tuple(row['xs']), tuple(row['ys'])), 0),
            axis=1
        )
        src[mode] = ColumnDataSource(df[mode])


        # Add hover tool
        hover[mode] = HoverTool(
            tooltips=[ # first string is the text
                ("", """ 
                ClusterID: @clids, Frac: @fracs{0.000}, FracSum: @fracs_sum{0.000}, En: @energies, EnSum: @energies_sum
                """),
            ],
            mode="mouse",
        )

        # Create figure
        p[mode] = [createFigure(title="Categorical distribution (" + mode + ")"),
                   createFigure(title="Continuous distribution (" + mode + ")")]
    
        # categorical figures
        colors = [Category10[10][i % 10] for i in df[mode].clids]
        src[mode].add(colors, "colors")
        p[mode][0].patches(
            xs="xs", ys="ys",
            source=src[mode],
            view=view[mode],
            fill_color="colors",
            line_color="black",
            fill_alpha=0.5,
        )
        
        # continuous figures
        mapper = LogColorMapper(palette=Viridis256, low=df[mode]['energies_sum'].min(), high=df[mode]['energies_sum'].max())
        color_bar = ColorBar(color_mapper=mapper, label_standoff=12,
                             title=("PF" if mode == "Reco" else "Sim") + " RecHit Energy Sum [GeV]",)
        p[mode][1].patches(
            xs="xs", ys="ys",
            source=src[mode],
            view=view[mode],
            fill_color=log_cmap('energies_sum', Viridis256, df[mode]['energies_sum'].min(), df[mode]['energies_sum'].max()),
            line_color="black",
        )
        p[mode][1].add_layout(color_bar, "right")
        
        # Add clusters
        # p[mode][1].scatter(
        #     x='clusterEtas' + mode,
        #     y='clusterPhis' + mode,
        #     source=srcCluster[mode],
        #     view=view[mode],
        #     color="red",
        #     marker="x",
        #     size=10,
        #     line_width=2,
        #     legend_label="Clusters",
        # )
        
        for idx in range(2):
            p[mode][idx].add_tools(hover[mode])

    p['Sim'][0].x_range, p['Sim'][0].y_range = p['Reco'][0].x_range, p['Reco'][0].y_range
    p['Sim'][1].x_range, p['Sim'][1].y_range = p['Reco'][0].x_range, p['Reco'][0].y_range
    p['Reco'][1].x_range, p['Reco'][1].y_range = p['Reco'][0].x_range, p['Reco'][0].y_range

    dfMin = min(df[mode].eventId.min() for mode in modes)
    dfMax = max(df[mode].eventId.max() for mode in modes)

    numInput = NumericInput(value=1, low=int(dfMin), high=int(dfMax),
                            title=f"Enter a number between {dfMin} and {dfMax}:")

    numInput_callb = CustomJS(args=dict(
        srcSim=src["Sim"], srcReco=src["Reco"],
        srcClSim=srcCluster["Sim"], srcClReco=srcCluster["Reco"],
        viewSim=view["Sim"], viewReco=view["Reco"],
        select=numInput
    ), code="""
    const eid = select.value.toString();
    viewSim.filters[0].group = eid;
    viewReco.filters[0].group = eid;
    viewSim.change.emit();
    viewReco.change.emit();
    srcSim.change.emit();
    srcReco.change.emit();
    srcClSim.change.emit();
    srcClReco.change.emit();
    """)
    numInput.js_on_change("value", numInput_callb)

    slider = Slider(start=0, end=1, value=0.1, step=0.01, title="Min Value")

    menu = [('Energy Sum', 'energies_sum'), ('Energy', 'energies'),
            ('Fraction Sum', 'fracs_sum'), ('Fraction', 'fracs')]
    dropdown = Dropdown(label="Z axis", button_type="warning", menu=menu, width=150)
    
    slider_calb = CustomJS(
        args=dict(
            srcSim=src["Sim"], srcReco=src["Reco"],
            viewSim=view["Sim"], viewReco=view["Reco"],
            threshSim=threshFilter["Sim"], threshReco=threshFilter["Reco"],
            slider=slider, select=numInput,
            varName="energies_sum"  # default
        ),
        code="""
        const minVal = slider.value;
        const eid = select.value.toString();
        
        const sim = srcSim.data;
        const rec = srcReco.data;
        
        let maskSim = [];
        let maskRec = [];
        
        for (let i = 0; i < sim[varName].length; i++) {
        maskSim.push(sim[varName][i] >= minVal && sim["eventId"][i] === eid);
        }
        for (let i = 0; i < rec[varName].length; i++) {
        maskRec.push(rec[varName][i] >= minVal && rec["eventId"][i] === eid);
        }
        
        threshSim.booleans = maskSim;
        threshReco.booleans = maskRec;
        
        srcSim.change.emit();
        srcReco.change.emit();
        """
    )
    slider.js_on_change("value", slider_calb)

    dropdown_calb = CustomJS(
        args=dict(
            srcSim=src["Sim"], srcReco=src["Reco"],
            patchSim=p["Sim"][1].renderers[0], patchReco=p["Reco"][1].renderers[0],
            mapperSim=p["Sim"][1].renderers[0].glyph.fill_color["transform"],
            mapperReco=p["Reco"][1].renderers[0].glyph.fill_color["transform"],
            slider=slider,
            slider_callback=slider_calb
        ),
        code="""
        const varName = this.item;
        // Update color mapper range
        mapperSim.low = Math.min(...srcSim.data[varName]);
        mapperSim.high = Math.max(...srcSim.data[varName]);
        mapperReco.low = Math.min(...srcReco.data[varName]);
        mapperReco.high = Math.max(...srcReco.data[varName]);
        // Update patch fill_color field
        patchSim.glyph.fill_color.field = varName;
        patchReco.glyph.fill_color.field = varName;
        // Update slider range and value
        slider.start = Math.min(...srcSim.data[varName],...srcReco.data[varName]);
        slider.end = Math.max(...srcSim.data[varName],...srcReco.data[varName]);
        slider.value = slider.start;
        slider.step = (slider.end - slider.start) / 100.;
        // Update the varName in the slider callback
        slider_callback.args.varName = varName;
        srcSim.change.emit();
        srcReco.change.emit();
        """
    )
    dropdown.js_on_event("menu_item_click", dropdown_calb)

    lay = layout([[numInput, slider, dropdown],
                  [p['Sim'][1], p['Reco'][1]],
                  [p['Sim'][0], p['Reco'][0]]])
    save(lay)
    print(f"INFO: Event plot saved to {output_path}")

def showECAL(infile, outfile, props, outname='EventDisplay'):
    varsGeom = [
        "crystalDetId",
        "crystalCenterEta",
        "crystalCenterPhi",
        "crystalCorner0Eta",
        "crystalCorner1Eta",
        "crystalCorner2Eta",
        "crystalCorner3Eta",
        "crystalCorner0Phi",
        "crystalCorner1Phi",
        "crystalCorner2Phi",
        "crystalCorner3Phi",
    ]
    varsEventCommon = ["eventId"]
    varsEvent = {"Reco": [], "Sim": []}
    for pfix in ("Reco", "Sim"):
        varsEvent[pfix].extend(
            [x + pfix for x in (
                "energies",
                "detids",
                "nHits",
                "clusterEnergies",
                "clusterEtas",
                "clusterPhis",
                "clusterHitEnergies",
                "clusterHitFractions",
                "clusterHitClids",
                "clusterHitDetids",
            )]
        )
    varsEventAll = varsEventCommon + varsEvent["Reco"] + varsEvent["Sim"]

    with uproot.open(infile) as file:
        dfGeom = file["ecalGeometryAnalyzer/Geometry"].arrays(varsGeom, library="pandas")
        dfEvent = file["ecalGeometryAnalyzer/Event"].arrays(varsEventAll, library="awkward")
    
    plotGeom(dfGeom, output_path=os.path.join(outfile, "geom.html"))

    dfHits, dfClusters, dfHitsInClusters = ({} for _ in range(3))
    for pfix in ("Reco", "Sim"):
        dfHits[pfix] = ak.to_dataframe(
            dfEvent[["eventId", "energies"+pfix, "detids"+pfix]]
        ).rename(columns={'energies'+pfix: 'energies', 'detids'+pfix: 'detids'})
        dfClusters[pfix] = ak.to_dataframe(dfEvent[["eventId", "clusterEnergies"+pfix, "clusterEtas"+pfix, "clusterPhis"+pfix]])
        dfHitsInClusters[pfix] = ak.to_dataframe(
            dfEvent[["eventId", "clusterHitEnergies"+pfix, "clusterHitDetids"+pfix, "clusterHitFractions"+pfix, "clusterHitClids"+pfix]]
        ).rename(columns={"clusterHitEnergies"+pfix: 'energies', "clusterHitDetids"+pfix: 'detids',
                          "clusterHitFractions"+pfix: 'fracs', "clusterHitClids"+pfix: 'clids'})

    plotEvent(
        dfGeom,
        dfHits,
        dfClusters,
        dfHitsInClusters,
        output_path=os.path.join(outfile, args.outname + ".html"),
        variables=("energy", "frac"),
    )

    print("INFO: Done.")

@dataclass
class InputArgs:
    nevents: int

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show position of crystals.")
    parser.add_argument("-i", "--file", help="Path to the input ROOT file.")
    parser.add_argument("-o", "--outdir", help="Path to the output folder where the script outputs will be stored.")
    parser.add_argument("--outname", default='EventDisplay', help="Name of the output html file with the event display.")
    parser.add_argument("-n", "--nevents", help="Number of events to plot.", default=6, type=int)

    args = parser.parse_args()
    props = InputArgs(nevents=args.nevents)
    showECAL(args.file, args.outdir, props, args.outname)
