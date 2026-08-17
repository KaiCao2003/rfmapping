import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const BUILD_DIR = "/Users/vxf1610/Developer/rfmapping/.codex_tmp/rf_population_deck_20260813";
const ASSET_DIR = "/Users/vxf1610/Developer/rfmapping/.codex_tmp/notebook_assets/inventory_5yECx2";
const FINAL_PPTX = "/Users/vxf1610/Developer/rfmapping/RF_population_HD_correspondence.pptx";

const W = 1280;
const H = 720;
const FONT = "Helvetica Neue";
const C = {
  canvas: "#FFFFFF",
  ink: "#000000",
  muted: "#5B6472",
  subtle: "#8A929F",
  panel: "#EDEDED",
  panelSoft: "#F6F7F8",
  rule: "#B8BCC4",
  accent: "#6DCBF4",
  accentStrong: "#3D8DFF",
  accentPale: "#D0EDFA",
  caution: "#B45309",
  cautionPale: "#FFF4E5",
};

const pt = (value) => value * 96 / 72;

async function readImageBlob(fileName) {
  const bytes = await fs.readFile(path.join(ASSET_DIR, fileName));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addText(slide, text, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: options.name,
    position,
    fill: options.fill ?? "none",
    line: options.line ?? { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: options.fontSize ?? pt(17),
    typeface: FONT,
    color: options.color ?? C.ink,
    bold: options.bold ?? false,
    italic: options.italic ?? false,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: options.autoFit ?? "shrinkText",
    wrap: options.wrap ?? "square",
    lineSpacing: options.lineSpacing,
    insets: options.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}

function addRect(slide, position, options = {}) {
  return slide.shapes.add({
    geometry: options.geometry ?? "rect",
    name: options.name,
    position,
    fill: options.fill ?? C.panel,
    line: options.line ?? { style: "solid", fill: "none", width: 0 },
    borderRadius: options.borderRadius,
  });
}

function addRule(slide, left, top, width, fill = C.rule, weight = 1) {
  return slide.shapes.add({
    geometry: "line",
    position: { left, top, width, height: 0 },
    fill: "none",
    line: { style: "solid", fill, width: weight },
  });
}

function addTitle(slide, title, section, page) {
  addText(slide, section.toUpperCase(), { left: 52, top: 28, width: 470, height: 22 }, {
    name: `section-${page}`,
    fontSize: pt(11),
    bold: true,
    color: C.muted,
    autoFit: "none",
  });
  addText(slide, title, { left: 52, top: 56, width: 1176, height: 64 }, {
    name: `title-${page}`,
    fontSize: pt(35),
    bold: true,
    autoFit: "shrinkText",
  });
  addRule(slide, 52, 666, 1176, C.rule, 1);
  addText(slide, "m15 · session 260630 · aggregate population analysis", { left: 52, top: 677, width: 700, height: 18 }, {
    name: `footer-${page}`,
    fontSize: pt(10),
    color: C.subtle,
    autoFit: "none",
  });
  addText(slide, String(page), { left: 1178, top: 677, width: 50, height: 18 }, {
    name: `page-${page}`,
    fontSize: pt(10),
    alignment: "right",
    color: C.subtle,
    autoFit: "none",
  });
}

function setSources(slide, lines) {
  slide.speakerNotes.textFrame.setText([
    "[Sources]",
    ...lines.map((line) => `- ${line}`),
    "[/Sources]",
  ]);
  slide.speakerNotes.setVisible(true);
}

function addImageFrame(slide, blob, alt, position, options = {}) {
  addRect(slide, position, {
    fill: options.frameFill ?? C.canvas,
    line: { style: "solid", fill: options.frameLine ?? C.rule, width: 1 },
  });
  return slide.images.add({
    blob,
    contentType: "image/png",
    alt,
    fit: options.fit ?? "contain",
    position,
    crop: options.crop,
  });
}

function addMetric(slide, x, y, width, number, label, detail, accent = C.accentStrong) {
  addRect(slide, { left: x, top: y, width, height: 5 }, { fill: accent });
  addText(slide, number, { left: x, top: y + 24, width, height: 68 }, {
    fontSize: pt(38),
    bold: true,
    autoFit: "none",
  });
  addText(slide, label, { left: x, top: y + 100, width, height: 34 }, {
    fontSize: pt(20),
    bold: true,
  });
  addText(slide, detail, { left: x, top: y + 144, width, height: 76 }, {
    fontSize: pt(15),
    color: C.muted,
  });
}

function addCallout(slide, x, y, width, heading, body, accent = C.accentStrong) {
  addRect(slide, { left: x, top: y, width: 5, height: 104 }, { fill: accent });
  addText(slide, heading, { left: x + 20, top: y, width: width - 20, height: 38 }, {
    fontSize: pt(24),
    bold: true,
  });
  addText(slide, body, { left: x + 20, top: y + 46, width: width - 20, height: 58 }, {
    fontSize: pt(15),
    color: C.muted,
  });
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  const [coverHeatmap, coverageSmoothed, centersSmoothed, sameUnitScatter, pairwiseDensity] = await Promise.all([
    readImageBlob("rf_profile_heatmap_sorted_by_hd_peak.png"),
    readImageBlob("rf_coverage_all_smoothed.png"),
    readImageBlob("rf_centers_all_smoothed.png"),
    readImageBlob("hd_same_unit_angle_alignment.png"),
    readImageBlob("hd_rf_pairwise_distance_density.png"),
  ]);

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  // Slide 1 — cover, following the Codex Grid half-text / image-field hierarchy.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addText(slide, "RF POPULATION ANALYSIS", { left: 52, top: 42, width: 500, height: 22 }, {
      fontSize: pt(11), bold: true, color: C.muted, autoFit: "none",
    });
    addText(slide, "Population RF localization and head-direction correspondence", { left: 52, top: 112, width: 570, height: 300 }, {
      name: "cover-title",
      fontSize: pt(50),
      bold: true,
      autoFit: "shrinkText",
    });
    addText(slide, "Aggregate results from two notebooks; Probes A and B", { left: 52, top: 442, width: 520, height: 72 }, {
      fontSize: pt(21),
      color: C.muted,
    });
    addText(slide, "m15 · session date 260630", { left: 52, top: 570, width: 450, height: 30 }, {
      fontSize: pt(16),
      bold: true,
    });
    addRect(slide, { left: 658, top: 42, width: 570, height: 588 }, { fill: C.panelSoft, line: { style: "solid", fill: C.rule, width: 1 } });
    slide.images.add({
      blob: coverHeatmap,
      contentType: "image/png",
      alt: "Population RF profiles for 57 units, ordered by HD peak",
      fit: "cover",
      position: { left: 690, top: 74, width: 506, height: 500 },
      crop: { left: 0.07, top: 0.0, right: 0.02, bottom: 0.0 },
    });
    addText(slide, "57-unit population RF profiles · HD-peak row order", { left: 690, top: 590, width: 506, height: 24 }, {
      fontSize: pt(11), color: C.muted, alignment: "right", autoFit: "none",
    });
    addText(slide, "1", { left: 1178, top: 677, width: 50, height: 18 }, {
      fontSize: pt(10), alignment: "right", color: C.subtle, autoFit: "none",
    });
    setSources(slide, [
      "/Users/vxf1610/Developer/rfmapping/hd_rf_comparison.ipynb, cell 8 output 1 (aggregate RF-profile heatmap).",
      "/Users/vxf1610/Developer/rfmapping/locate_rf.ipynb and hd_rf_comparison.ipynb (deck scope and session identifiers).",
    ]);
  }

  // Slide 2 — linked questions, using the paired narrative silhouette.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addTitle(slide, "Two analyses address one population question", "Framing", 2);
    addText(slide, "Where are visually responsive fields located, and does their azimuth align with the same neurons’ head-direction preference?", { left: 52, top: 142, width: 1176, height: 56 }, {
      fontSize: pt(20),
      color: C.muted,
    });
    addRule(slide, 640, 228, 0, C.rule, 1);
    addRect(slide, { left: 52, top: 226, width: 548, height: 326 }, { fill: C.panelSoft });
    addText(slide, "01", { left: 78, top: 252, width: 58, height: 36 }, { fontSize: pt(18), bold: true, color: C.accentStrong });
    addText(slide, "Locate population RFs", { left: 142, top: 248, width: 410, height: 44 }, { fontSize: pt(25), bold: true });
    addText(slide, "Detect significant ON-response clusters for each retained unit, then aggregate full masks and one response-weighted center per detected unit.", { left: 78, top: 322, width: 474, height: 110 }, { fontSize: pt(17), color: C.muted });
    addText(slide, "Outputs", { left: 78, top: 458, width: 120, height: 28 }, { fontSize: pt(15), bold: true });
    addText(slide, "Population overlap · center density · horizontal and vertical projections", { left: 78, top: 492, width: 474, height: 54 }, { fontSize: pt(15), color: C.muted });

    addRect(slide, { left: 680, top: 226, width: 548, height: 326 }, { fill: C.panelSoft });
    addText(slide, "02", { left: 706, top: 252, width: 58, height: 36 }, { fontSize: pt(18), bold: true, color: C.accentStrong });
    addText(slide, "Compare HD and RF angles", { left: 770, top: 248, width: 410, height: 44 }, { fontSize: pt(25), bold: true });
    addText(slide, "Match units across sessions, compare HD peak with RF-center azimuth, and test whether relative angular spacing is preserved across the cohort.", { left: 706, top: 322, width: 474, height: 110 }, { fontSize: pt(17), color: C.muted });
    addText(slide, "Outputs", { left: 706, top: 458, width: 120, height: 28 }, { fontSize: pt(15), bold: true });
    addText(slide, "Circular correlation · RF-minus-HD offset · Spearman Mantel statistic", { left: 706, top: 492, width: 474, height: 54 }, { fontSize: pt(15), color: C.muted });

    addText(slide, "Common coordinate", { left: 52, top: 594, width: 210, height: 28 }, { fontSize: pt(15), bold: true });
    addText(slide, "RF horizontal position is mapped onto the HD convention as mod(−x, 360°).", { left: 270, top: 594, width: 850, height: 30 }, { fontSize: pt(15), color: C.muted });
    setSources(slide, [
      "/Users/vxf1610/Developer/rfmapping/locate_rf.ipynb, cells 2, 6–8.",
      "/Users/vxf1610/Developer/rfmapping/hd_rf_comparison.ipynb, cells 4–11.",
    ]);
  }

  // Slide 3 — cohort flow, following the Codex Grid timeline silhouette.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addTitle(slide, "268 units narrow to 57 matched HD–RF units", "Cohort", 3);
    addText(slide, "Each stage answers a different question; denominators should not be interchanged.", { left: 52, top: 140, width: 800, height: 36 }, { fontSize: pt(18), color: C.muted });
    addRule(slide, 110, 302, 1010, C.rule, 2);
    const nodes = [
      { x: 110, number: "268", label: "Raw RF units", detail: "A 146 · B 122", fill: C.ink },
      { x: 435, number: "241", label: "Retained", detail: "A 134 · B 107\n89.9% of raw", fill: C.accentStrong },
      { x: 760, number: "64", label: "Detected RF", detail: "A 37 · B 27\n26.6% of retained", fill: C.accentStrong },
      { x: 1085, number: "57", label: "Matched HD + RF", detail: "HD class 2 + shared ID\n+ detected RF center", fill: C.ink },
    ];
    for (const node of nodes) {
      slide.shapes.add({ geometry: "ellipse", position: { left: node.x - 9, top: 293, width: 18, height: 18 }, fill: node.fill, line: { style: "solid", fill: node.fill, width: 0 } });
      addText(slide, node.number, { left: node.x - 86, top: 208, width: 172, height: 62 }, { fontSize: pt(40), bold: true, alignment: "center", autoFit: "none" });
      addText(slide, node.label, { left: node.x - 130, top: 338, width: 260, height: 42 }, { fontSize: pt(20), bold: true, alignment: "center" });
      addText(slide, node.detail, { left: node.x - 140, top: 394, width: 280, height: 68 }, { fontSize: pt(15), color: C.muted, alignment: "center" });
    }
    addRect(slide, { left: 52, top: 514, width: 1176, height: 108 }, { fill: C.panelSoft });
    addText(slide, "What changes at the final step?", { left: 78, top: 540, width: 310, height: 32 }, { fontSize: pt(18), bold: true });
    addText(slide, "The 57-unit correspondence cohort is selected for strong HD tuning, a shared unit ID across sessions, and a significant RF center. It is not an estimate across all recorded units.", { left: 400, top: 536, width: 780, height: 62 }, { fontSize: pt(16), color: C.muted });
    setSources(slide, [
      "/Users/vxf1610/Developer/rfmapping/locate_rf.ipynb, cells 2–3 (raw, retained, and detected-unit counts).",
      "/Users/vxf1610/Developer/rfmapping/hd_rf_comparison.ipynb, cells 4 and 7 (57-unit selected cohort).",
    ]);
  }

  // Slide 4 — method timeline plus the two material caveats.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addTitle(slide, "RF detection is rigorous; selection and timing matter", "Method", 4);
    addRule(slide, 92, 300, 1096, C.rule, 2);
    const steps = [
      { x: 92, tag: "INPUT", heading: "Reconstruct ON trials", body: "10,500 ON presentations across a 7 × 30 grid; 50 repeats per position." },
      { x: 480, tag: "TEST", heading: "Detect positive clusters", body: "Count spikes in [0, 0.2) s; z ≥ 1.5; 10,000 block-preserving permutations; α = .05." },
      { x: 868, tag: "OUTPUT", heading: "Aggregate masks and centers", body: "Retain full significant masks and one response-weighted discrete center per detected unit." },
    ];
    for (const step of steps) {
      slide.shapes.add({ geometry: "ellipse", position: { left: step.x - 8, top: 292, width: 16, height: 16 }, fill: C.accentStrong, line: { style: "solid", fill: C.accentStrong, width: 0 } });
      addText(slide, step.tag, { left: step.x, top: 212, width: 150, height: 24 }, { fontSize: pt(11), bold: true, color: C.accentStrong });
      addText(slide, step.heading, { left: step.x, top: 334, width: 286, height: 42 }, { fontSize: pt(20), bold: true });
      addText(slide, step.body, { left: step.x, top: 386, width: 286, height: 100 }, { fontSize: pt(15), color: C.muted });
    }
    addRect(slide, { left: 52, top: 520, width: 1176, height: 116 }, { fill: C.cautionPale });
    addText(slide, "Cohort screen", { left: 78, top: 546, width: 170, height: 30 }, { fontSize: pt(17), bold: true, color: C.caution });
    addText(slide, "Any zero pooled-count spatial bin excludes a unit, favoring broadly active units.", { left: 78, top: 584, width: 490, height: 42 }, { fontSize: pt(14), color: C.muted });
    addRule(slide, 620, 542, 0, C.rule, 1);
    addText(slide, "Timing overlap", { left: 662, top: 546, width: 170, height: 30 }, { fontSize: pt(17), bold: true, color: C.caution });
    addText(slide, "Stimuli are ≈100 ms apart, so a 200 ms response window usually extends beyond the next onset.", { left: 662, top: 584, width: 490, height: 42 }, { fontSize: pt(14), color: C.muted });
    setSources(slide, [
      "/Users/vxf1610/Developer/rfmapping/locate_rf.ipynb, cells 1–2 (window, z threshold, filtering, and RF detection).",
      "Remote source-of-truth session data: /mnt/senzailab/Kai/#Recording/m15/260630 on hhw9l84 (trial count and onset spacing).",
      "Remote MATLAB generator context: /mnt/ssd4.1/Matlab on hhw9l84 (7 × 30 visual grid and repeats).",
    ]);
  }

  // Slide 5 — population localization maps.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addTitle(slide, "Detected RFs form a structured population map", "RF localization", 5);
    addText(slide, "Both displays combine Probes A and B. Gaussian smoothing (σ = 1 grid bin) is used only to reveal population structure.", { left: 52, top: 138, width: 1120, height: 40 }, { fontSize: pt(16), color: C.muted });

    addText(slide, "Full significant-mask overlap", { left: 52, top: 198, width: 530, height: 34 }, { fontSize: pt(21), bold: true });
    addImageFrame(slide, coverageSmoothed, "Smoothed combined population overlap of full significant RF masks", { left: 52, top: 240, width: 548, height: 272 }, {
      fit: "cover",
      crop: { left: 0.02, top: 0.27, right: 0.02, bottom: 0.25 },
    });
    addText(slide, "Coverage emphasizes the spatial extent shared by detected RF masks.", { left: 52, top: 526, width: 548, height: 42 }, { fontSize: pt(15), color: C.muted });
    addText(slide, "Peak raw overlap: 18 units", { left: 52, top: 584, width: 548, height: 32 }, { fontSize: pt(20), bold: true, color: C.accentStrong });

    addText(slide, "Response-weighted RF-center density", { left: 680, top: 198, width: 548, height: 34 }, { fontSize: pt(21), bold: true });
    addImageFrame(slide, centersSmoothed, "Smoothed combined density of one response-weighted RF center per detected unit", { left: 680, top: 240, width: 548, height: 272 }, {
      fit: "cover",
      crop: { left: 0.02, top: 0.27, right: 0.02, bottom: 0.25 },
    });
    addText(slide, "Centers collapse every detected unit to one discrete grid bin and expose multiple modes.", { left: 680, top: 526, width: 548, height: 42 }, { fontSize: pt(15), color: C.muted });
    addText(slide, "Peak raw center bin: 6 units", { left: 680, top: 584, width: 548, height: 32 }, { fontSize: pt(20), bold: true, color: C.accentStrong });
    setSources(slide, [
      "/Users/vxf1610/Developer/rfmapping/locate_rf.ipynb, cell 6 output 6 (combined smoothed RF-mask overlap).",
      "/Users/vxf1610/Developer/rfmapping/locate_rf.ipynb, cell 7 output 6 (combined smoothed RF-center density).",
      "Raw aggregate arrays validated from the same notebook/session for peak overlap and center-bin counts.",
    ]);
  }

  // Slide 6 — redraw ambiguous row/column projections as editable degree-coordinate charts.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addTitle(slide, "RF centers cluster at y = −15° and several azimuths", "RF localization", 6);
    const xPositions = Array.from({ length: 30 }, (_, i) => -174 + i * 12);
    const xCounts = [0, 0, 1, 0, 1, 0, 4, 1, 7, 8, 10, 6, 1, 0, 1, 0, 1, 0, 0, 0, 8, 3, 1, 1, 7, 1, 2, 0, 0, 0];
    const yPositions = [-39, -27, -15, -3, 9, 21, 33];
    const yCounts = [13, 5, 26, 13, 0, 3, 4];

    addText(slide, "Horizontal center distribution", { left: 52, top: 148, width: 716, height: 34 }, { fontSize: pt(21), bold: true });
    slide.charts.add("scatter", {
      position: { left: 42, top: 190, width: 750, height: 360 },
      series: [{
        name: "RF centers",
        xValues: xPositions,
        values: xCounts,
        line: { style: "solid", width: 3, fill: C.accentStrong },
        marker: { symbol: "circle", size: 6 },
        fill: C.accentStrong,
      }],
      hasLegend: false,
      chartFill: C.canvas,
      chartLine: { style: "solid", width: 0, fill: C.canvas },
      plotAreaFill: { type: "none" },
      plotAreaLine: { style: "solid", width: 0, fill: C.canvas },
      scatterOptions: { style: "lineWithMarkers", varyColors: false },
      xAxis: {
        visible: true,
        title: { text: "Visual azimuth (deg)", textStyle: { fontSize: 18, fill: C.ink } },
        min: -180,
        max: 180,
        majorUnit: 60,
        textStyle: { fontSize: 15, fill: C.muted },
        line: { style: "solid", width: 1, fill: C.rule },
        majorGridlines: { style: "solid", width: 1, fill: C.panel },
      },
      yAxis: {
        visible: true,
        title: { text: "RF centers (units)", textStyle: { fontSize: 18, fill: C.ink } },
        min: 0,
        max: 12,
        majorUnit: 2,
        textStyle: { fontSize: 15, fill: C.muted },
        majorGridlines: { style: "solid", width: 1, fill: C.panel },
        line: { style: "solid", width: 0, fill: C.canvas },
      },
    });
    addText(slide, "Main azimuthal modes", { left: 66, top: 562, width: 210, height: 26 }, { fontSize: pt(14), bold: true });
    addText(slide, "−54°: 10 centers   ·   +66°: 8   ·   +114°: 7", { left: 290, top: 560, width: 480, height: 30 }, { fontSize: pt(14), color: C.muted });

    addRect(slide, { left: 828, top: 144, width: 400, height: 472 }, { fill: C.panelSoft });
    addText(slide, "Vertical center distribution", { left: 856, top: 168, width: 344, height: 34 }, { fontSize: pt(21), bold: true });
    slide.charts.add("bar", {
      position: { left: 844, top: 214, width: 368, height: 280 },
      categories: yPositions.map((value) => `${value}°`),
      series: [{ name: "RF centers", categories: yPositions.map((value) => `${value}°`), values: yCounts, fill: C.accent }],
      hasLegend: false,
      chartFill: C.panelSoft,
      chartLine: { style: "solid", width: 0, fill: C.panelSoft },
      plotAreaFill: { type: "none" },
      plotAreaLine: { style: "solid", width: 0, fill: C.panelSoft },
      barOptions: { direction: "bar", grouping: "clustered", gapWidth: 45 },
      xAxis: {
        visible: true,
        min: 0,
        max: 30,
        majorUnit: 5,
        textStyle: { fontSize: 14, fill: C.muted },
        majorGridlines: { style: "solid", width: 1, fill: C.rule },
        line: { style: "solid", width: 1, fill: C.rule },
      },
      yAxis: {
        visible: true,
        textStyle: { fontSize: 15, fill: C.muted },
        line: { style: "solid", width: 0, fill: C.panelSoft },
      },
      dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 14, fill: C.ink, bold: true } },
    });
    addText(slide, "26 / 64", { left: 856, top: 520, width: 180, height: 54 }, { fontSize: pt(32), bold: true, color: C.accentStrong, autoFit: "none" });
    addText(slide, "centers fall at y = −15°", { left: 1014, top: 532, width: 178, height: 40 }, { fontSize: pt(14), color: C.muted });
    setSources(slide, [
      "/Users/vxf1610/Developer/rfmapping/locate_rf.ipynb, cell 8 (horizontal and vertical center-count projections).",
      "Remote generator/session validation on hhw9l84 for degree coordinates of the 7 × 30 grid.",
    ]);
  }

  // Slide 7 — same-unit correspondence.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addTitle(slide, "HD preference tracks RF azimuth in selected units", "HD–RF correspondence", 7);
    addImageFrame(slide, sameUnitScatter, "Scatter of HD peak angle against detected 2-D RF center angle for 57 matched units", { left: 52, top: 148, width: 650, height: 494 }, { fit: "contain" });
    addCallout(slide, 758, 164, 430, "ρ = 0.521", "Circular–circular correlation; permutation p < 10⁻⁴.", C.accentStrong);
    addRule(slide, 758, 294, 430, C.rule, 1);
    addCallout(slide, 758, 322, 430, "+6.1° offset", "Mean RF-minus-HD angular offset; alignment R = 0.658.", C.accentStrong);
    addRule(slide, 758, 452, 430, C.rule, 1);
    addCallout(slide, 758, 480, 430, "n = 57", "HD-class-2 units with a detected RF center and a shared unit ID across sessions.", C.ink);
    setSources(slide, [
      "/Users/vxf1610/Developer/rfmapping/hd_rf_comparison.ipynb, cell 7 output 0 and same_unit_statistics.",
      "Permutation p = 1/10,001 is the minimum attainable value with 10,000 resamples; displayed as p < 10⁻⁴.",
    ]);
  }

  // Slide 8 — pairwise organization.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addTitle(slide, "Relative angular spacing is preserved across units", "HD–RF correspondence", 8);
    addImageFrame(slide, pairwiseDensity, "Density of pairwise circular HD-peak distances against RF-center distances", { left: 52, top: 146, width: 650, height: 500 }, { fit: "contain" });
    addCallout(slide, 758, 164, 430, "ρ = 0.515", "Spearman Mantel association between pairwise circular distances; p < 10⁻⁴.", C.accentStrong);
    addRule(slide, 758, 294, 430, C.rule, 1);
    addCallout(slide, 758, 322, 430, "1,596 pairs", "Unique off-diagonal unit pairs used by the inferential statistic.", C.accentStrong);
    addRule(slide, 758, 452, 430, C.rule, 1);
    addCallout(slide, 758, 480, 430, "3,249 entries", "The density image shows 57² ordered entries, including self-pairs and symmetric duplicates.", C.ink);
    setSources(slide, [
      "/Users/vxf1610/Developer/rfmapping/hd_rf_comparison.ipynb, cell 11 output 0 and pairwise_statistics.",
      "Unique-pair count is 57 × 56 / 2 = 1,596; the displayed density uses 57² = 3,249 ordered entries.",
    ]);
  }

  // Slide 9 — synthesis and limits.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addTitle(slide, "Population data link visual and directional coding", "Conclusion", 9);
    addText(slide, "Best current interpretation", { left: 52, top: 144, width: 420, height: 38 }, { fontSize: pt(24), bold: true });
    addText(slide, "Within a selected single-session cohort, visual RF azimuth and head-direction preference share both same-unit alignment and population geometry.", { left: 52, top: 192, width: 1120, height: 76 }, { fontSize: pt(22), color: C.muted });
    addMetric(slide, 52, 304, 326, "64 / 241", "RF localization", "26.6% of retained units had a detected ON-response RF cluster.", C.accent);
    addMetric(slide, 460, 304, 326, "57 units", "Matched cohort", "Selected for HD class 2, shared ID, and a detected RF center.", C.accentStrong);
    addMetric(slide, 868, 304, 326, "ρ ≈ 0.52", "Convergent structure", "Same-unit circular and pairwise Mantel associations are both moderate.", C.ink);
    addRect(slide, { left: 52, top: 550, width: 1176, height: 92 }, { fill: C.cautionPale });
    addText(slide, "Interpret as exploratory population evidence", { left: 78, top: 576, width: 380, height: 32 }, { fontSize: pt(18), bold: true, color: C.caution });
    addText(slide, "One animal/date; selected units; per-unit spatial correction; discrete centers; and 200 ms windows that overlap the next stimulus. Replication with tighter timing is the decisive next test.", { left: 470, top: 570, width: 700, height: 52 }, { fontSize: pt(14), color: C.muted });
    setSources(slide, [
      "/Users/vxf1610/Developer/rfmapping/locate_rf.ipynb, cells 2–8.",
      "/Users/vxf1610/Developer/rfmapping/hd_rf_comparison.ipynb, cells 4–11.",
      "Remote source-of-truth session timing on hhw9l84 for the response-window overlap caveat.",
    ]);
  }

  await fs.mkdir(path.join(BUILD_DIR, "preview"), { recursive: true });
  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(BUILD_DIR, "preview", `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(BUILD_DIR, "preview", `${stem}.layout.json`), await layout.text());
  }
  await writeBlob(path.join(BUILD_DIR, "preview", "deck-montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));
  const snapshot = await presentation.inspect({ kind: "slide,textbox,shape,image,chart,notes", maxChars: 50000 });
  await fs.writeFile(path.join(BUILD_DIR, "preview", "deck-inspect.ndjson"), snapshot.ndjson);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(FINAL_PPTX);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
