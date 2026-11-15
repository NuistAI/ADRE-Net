'use strict';

const MAP_PIXELS = RES * RES;           // 224*224
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const origCanvas = document.getElementById('origCanvas');
const gtCanvas = document.getElementById('gtCanvas');
const colCanvas = document.getElementById('colCanvas');
const depthCanvas = document.getElementById('depthCanvas');
const colorbar = document.getElementById('colorbar');

const origCtx = origCanvas.getContext('2d');
const gtCtx = gtCanvas.getContext('2d');
const colCtx = colCanvas.getContext('2d');
const depthCtx = depthCanvas.getContext('2d');
const cbCtx = colorbar.getContext('2d');

const hcmsOrigCanvas = document.getElementById('hcmsOrigCanvas');
const hcmsGtCanvas = document.getElementById('hcmsGtCanvas');
const hAttnCanvas = document.getElementById('hAttnCanvas');
const hcmsOrigCtx = hcmsOrigCanvas.getContext('2d');
const hcmsGtCtx = hcmsGtCanvas.getContext('2d');
const hAttnCtx = hAttnCanvas.getContext('2d');
const hSpatialCanvas = document.getElementById('hSpatialCanvas');
const hSpatialCtx = hSpatialCanvas.getContext('2d');
const hColorbar = document.getElementById('hColorbar');
const hCbCtx = hColorbar.getContext('2d');

let origImg, gtImg;
let rowMaps, colMaps;                   // Uint8Array
let gtData;                             // ImageData.data for GT strip extraction
let curDataset = 'nAMD';
const DATASETS = {};                    // key -> {origImg, gtImg, rowMaps, colMaps, gtData}
let curCol = DEFAULT_COL;
let curRow = DEFAULT_ROW;
let pinnedCol = DEFAULT_COL;
let pinnedRow = DEFAULT_ROW;
let locked = false;   // 图1: 单击后锁定, 不再跟随鼠标
let lockedCol = false; // 图3: 单击确定后锁定, 不再移动
let lockedDepth = false; // 图4: 单击确定后锁定, 不再左右滑动

let hAttnU8 = null;     // 水平注意力矩阵 [224*224] uint8
let hAttnColorCache = null; // 注意力矩阵 colorize 后的 ImageData 缓存(不随列变化)
let hGtStripH = null;   // GT 列色条(水平, 224×HSTRIP)
let hGtStripV = null;   // GT 列色条(垂直, HSTRIP×224)
let hOrigImg = null;    // 当前原图
let hGtImg = null;      // 当前 GT
let hCol = 112;         // 当前选中列
let hFixedCol = 112;    // 固定列
let hCurDataset = 'dme';
const HDATASETS = {};   // key -> {origImg, gtImg, attnU8}

// VIRIDIS 256-entry RGBA LUT
const virLUT = new Uint8Array(256 * 4);
for (let i = 0; i < 256; i++) {
  virLUT[i * 4]     = VIRIDIS[i * 3];
  virLUT[i * 4 + 1] = VIRIDIS[i * 3 + 1];
  virLUT[i * 4 + 2] = VIRIDIS[i * 3 + 2];
  virLUT[i * 4 + 3] = 255;
}

function loadImage(src) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = () => rej(new Error('load failed: ' + src));
    img.src = src;
  });
}

async function loadBin(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('fetch failed: ' + url);
  return new Uint8Array(await r.arrayBuffer());
}

// 加载 .gz 压缩的二进制文件并在浏览器端解压 (gzip)
async function loadBinGz(url) {
  const r = await fetch(url + '.gz');
  if (!r.ok) throw new Error('fetch failed: ' + url + '.gz');
  const ds = new DecompressionStream('gzip');
  const buf = await new Response(r.body.pipeThrough(ds)).arrayBuffer();
  return new Uint8Array(buf);
}

function getImgData(img, w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const cx = c.getContext('2d');
  cx.drawImage(img, 0, 0, w, h);
  return cx.getImageData(0, 0, w, h);
}

// 从 uint8 归一化图 (src[offset .. offset+MAP_PIXELS]) 生成 VIRIDIS 着色 ImageData
function colorize(src, offset) {
  const id = depthCtx.createImageData(RES, RES);
  const d = id.data;
  for (let i = 0; i < MAP_PIXELS; i++) {
    const v = src[offset + i];
    const vi = v * 4;
    d[i * 4]     = virLUT[vi];
    d[i * 4 + 1] = virLUT[vi + 1];
    d[i * 4 + 2] = virLUT[vi + 2];
    d[i * 4 + 3] = 255;
  }
  return id;
}

// 将单列注意力 row(长度 RES) 沿深度复制成整张热图并 VIRIDIS 着色(比先建 heat 再 colorize 快)
function colorizeHeat(row) {
  const id = depthCtx.createImageData(RES, RES);
  const d = id.data;
  const firstRow = new Uint8Array(RES * 4);
  for (let x = 0; x < RES; x++) {
    const vi = row[x] * 4;
    firstRow[x * 4]     = virLUT[vi];
    firstRow[x * 4 + 1] = virLUT[vi + 1];
    firstRow[x * 4 + 2] = virLUT[vi + 2];
    firstRow[x * 4 + 3] = 255;
  }
  for (let y = 0; y < RES; y++) d.set(firstRow, y * RES * 4);
  return id;
}

function drawHLine(ctx, y, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0.5, y + 0.5);
  ctx.lineTo(RES - 0.5, y + 0.5);
  ctx.stroke();
}

function drawVLine(ctx, x, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x + 0.5, 0.5);
  ctx.lineTo(x + 0.5, RES - 0.5);
  ctx.stroke();
}

// 箭头悬停透明度状态
let colHovered = false;     // IVSA 图3
let depthHovered = false;   // IVSA 图4
let hAttnHovered = false;   // IHSA 图3
let hOrigHovered = false;   // IHSA 图1
let hGtHovered = false;     // IHSA 图2
let hSpatialHovered = false;// IHSA 图4
let hLocked = false;        // IHSA 单击锁定(红线固定, 灰线跟随)
let hHoverCol = null;       // IHSA 悬停列(锁定后灰色线)
let hHoverFig = null;       // IHSA 当前悬停的图: orig/gt/attn/spatial
let gtHoverCol = null;      // IVSA 图2 悬停位置(列)
let gtHoverRow = null;      // IVSA 图2 悬停位置(行)
let origHoverCol = null;    // IVSA 图1 悬停位置(列, 单击锁定后)
let origHoverRow = null;    // IVSA 图1 悬停位置(行, 单击锁定后)
let colHoverRow = null;     // IVSA 图3 悬停行(单击锁定后, 灰色横线)
let depthHoverCol = null;   // IVSA 图4 悬停列(单击锁定后, 灰色竖线)
const ARROW_ALPHA_IDLE = 0.3;   // 默认更透明
const ARROW_ALPHA_HOVER = 0.8;  // 悬停更实

// 在线的右端上下各画一个箭头, 表示可上下滑动 (白底黑边, 半透明)
function drawUpDownArrows(ctx, x, y, color, alpha) {
  ctx.font = 'bold 16px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 3;
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = '#000000';
  ctx.strokeText('▲', x, y - 11);
  ctx.strokeText('▼', x, y + 11);
  ctx.fillStyle = color;
  ctx.fillText('▲', x, y - 11);
  ctx.fillText('▼', x, y + 11);
  ctx.globalAlpha = 1.0;
}

// 在线的下端左右各画一个箭头, 表示可左右滑动 (白底黑边, 半透明)
function drawLeftRightArrows(ctx, x, y, color, alpha) {
  ctx.font = 'bold 16px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 3;
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = '#000000';
  ctx.strokeText('◀', x - 11, y);
  ctx.strokeText('▶', x + 11, y);
  ctx.fillStyle = color;
  ctx.fillText('◀', x - 11, y);
  ctx.fillText('▶', x + 11, y);
  ctx.globalAlpha = 1.0;
}

// 从 GT 图提取列色条: 去掉整行都是背景的行, 高度压缩到 5px (224×5)
function computeGtStrip(gtImg) {
  const tmp = document.createElement('canvas');
  tmp.width = RES; tmp.height = RES;
  const tctx = tmp.getContext('2d');
  tctx.drawImage(gtImg, 0, 0);
  const d = tctx.getImageData(0, 0, RES, RES).data;

  let minRow = RES, maxRow = -1;
  for (let y = 0; y < RES; y++) {
    let allBg = true;
    for (let x = 0; x < RES; x++) {
      const o = (y * RES + x) * 4;
      const r = d[o], g = d[o + 1], b = d[o + 2];
      const isBg = (r === 170 && g === 170 && b === 170) || (r === 60 && g === 60 && b === 60);
      if (!isBg) { allBg = false; break; }
    }
    if (!allBg) {
      if (y < minRow) minRow = y;
      if (y > maxRow) maxRow = y;
    }
  }
  if (maxRow < 0) { minRow = 0; maxRow = RES - 1; }
  const cropH = maxRow - minRow + 1;

  const strip = document.createElement('canvas');
  strip.width = RES; strip.height = HSTRIP;
  const sctx = strip.getContext('2d');
  sctx.drawImage(gtImg, 0, minRow, RES, cropH, 0, 0, RES, HSTRIP);
  return strip;
}

// 将 224×HSTRIP 水平色条转置成 HSTRIP×224 垂直色条 (列 y 对应 GT 列 y, 列 0 在上)
function transposeStrip(hstrip) {
  const sdata = hstrip.getContext('2d').getImageData(0, 0, RES, HSTRIP).data;
  const vc = document.createElement('canvas');
  vc.width = HSTRIP; vc.height = RES;
  const vctx = vc.getContext('2d');
  const vd = vctx.createImageData(HSTRIP, RES);
  for (let y = 0; y < RES; y++) {
    for (let x = 0; x < HSTRIP; x++) {
      const so = (x * RES + y) * 4;   // 水平: 行=x(深度), 列=y(A-scan)
      const dst = (y * HSTRIP + x) * 4;    // 垂直: 行=y(A-scan), 列=x(深度)
      vd.data[dst]     = sdata[so];
      vd.data[dst + 1] = sdata[so + 1];
      vd.data[dst + 2] = sdata[so + 2];
      vd.data[dst + 3] = 255;
    }
  }
  vctx.putImageData(vd, 0, 0);
  return vc;
}

// 原图 / GT: 底图 + 红线(行) + 绿线(列) + 交点白圆点
function renderPanel(ctx, img, row, col) {
  ctx.clearRect(0, 0, RES, RES);
  ctx.drawImage(img, 0, 0);
  drawHLine(ctx, row, '#ff0000');
  drawVLine(ctx, col, '#00ff00');
  ctx.fillStyle = '#ffffff';
  ctx.beginPath(); ctx.arc(col, row, 6, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(col, row, 6, 0, Math.PI * 2); ctx.stroke();
}

// 第 3 列: 列注意力热图 + 左/上 GT 色条 + 绿框 + 红线(查询深度)
function renderColPanel(col, row) {
  const S = STRIP + RES;                      // 248
  const TS = STRIP / 2;                       // 上方条高度 = 12 (降为一半)
  colCtx.clearRect(0, 0, S, S);

  colCtx.putImageData(colorize(colMaps, col * MAP_PIXELS), STRIP - 4, STRIP);

  // GT 色条: 左(垂直, query 轴) + 上(水平, key 轴, 高 TS)
  for (let d = 0; d < RES; d++) {
    const o = (d * RES + col) * 4;
    colCtx.fillStyle = `rgb(${gtData[o]},${gtData[o + 1]},${gtData[o + 2]})`;
    colCtx.fillRect(0, STRIP + d, STRIP - 4, 1);        // 左条(Query, 宽度-2px)
    colCtx.fillRect(STRIP - 4 + d, TS, 1, TS);          // 上条 (高 12)
  }

  // 两条 GT 绿色边框
  colCtx.strokeStyle = '#00ff00';
  colCtx.lineWidth = 1;
  colCtx.strokeRect(0.5, STRIP + 0.5, STRIP - 5, RES - 1);       // 左条
  colCtx.strokeRect(STRIP - 4 + 0.5, TS + 0.5, RES - 1, TS - 1);     // 上条

  // 查询深度线: GT 框(左条)内红色, 热图内绿色
  const ly = STRIP + row + 0.5;
  colCtx.lineWidth = 1;
  // 红色段: 左 GT 条
  colCtx.strokeStyle = '#ff0000';
  colCtx.beginPath();
  colCtx.moveTo(0.5, ly);
  colCtx.lineTo(STRIP - 4 + 0.5, ly);
  colCtx.stroke();
  // 绿色段: 热图
  colCtx.strokeStyle = '#00ff00';
  colCtx.beginPath();
  colCtx.moveTo(STRIP - 4 + 0.5, ly);
  colCtx.lineTo(S - 4 - 0.5, ly);
  colCtx.stroke();

  // 上下箭头: 表示该线可上下滑动
  drawUpDownArrows(colCtx, S - 13, ly, '#ffffff', colHovered ? ARROW_ALPHA_HOVER : ARROW_ALPHA_IDLE);

  // 纵轴 GT(左条) 中标记当前像素点 (红线与左条交点)
  colCtx.fillStyle = '#ffffff';
  colCtx.beginPath(); colCtx.arc((STRIP - 4) / 2, STRIP + row, 3, 0, Math.PI * 2); colCtx.fill();
  colCtx.strokeStyle = '#000000';
  colCtx.lineWidth = 1;
  colCtx.beginPath(); colCtx.arc((STRIP - 4) / 2, STRIP + row, 3, 0, Math.PI * 2); colCtx.stroke();
}

// 第 4 列: 行注意力(深度) 叠加图 + 上方 GT 色条 + 绿竖线(列) + 红线/锚点
function renderDepthPanel(row, col) {
  const W = RES;                 // 224 (热图宽度)
  const H = STRIP + RES;         // 248 (画布高度)
  const TS = STRIP / 2;          // 12 (GT 条高度, 与图3 key 条一致)
  depthCtx.clearRect(0, 0, W, H);

  // 上方 GT 色条: 该行(红线对应)逐列 GT 颜色 (x=0..RES, 高 TS)
  for (let c = 0; c < RES; c++) {
    const o = (row * RES + c) * 4;
    depthCtx.fillStyle = `rgb(${gtData[o]},${gtData[o + 1]},${gtData[o + 2]})`;
    depthCtx.fillRect(c, TS, 1, TS);
  }
  depthCtx.strokeStyle = '#ff0000';
  depthCtx.lineWidth = 1;
  depthCtx.strokeRect(0.5, TS + 0.5, RES - 1, TS - 1);

  // 热图(原图 30% + 注意力 70%) 从 (0, STRIP) 开始
  depthCtx.putImageData(colorize(rowMaps, row * MAP_PIXELS), 0, STRIP);
  depthCtx.globalAlpha = 0.30;
  depthCtx.drawImage(origImg, 0, STRIP);
  depthCtx.globalAlpha = 1.0;

  // 红线: 查询深度(key_depth=row), 仅热图内 (加粗 + 黑色描边)
  const ry = STRIP + row + 0.5;
  depthCtx.strokeStyle = '#000000';
  depthCtx.lineWidth = 5;
  depthCtx.beginPath();
  depthCtx.moveTo(0.5, ry);
  depthCtx.lineTo(RES - 0.5, ry);
  depthCtx.stroke();
  depthCtx.strokeStyle = '#ff0000';
  depthCtx.lineWidth = 3;
  depthCtx.beginPath();
  depthCtx.moveTo(0.5, ry);
  depthCtx.lineTo(RES - 0.5, ry);
  depthCtx.stroke();

  // 绿色竖线: 当前点对应列 (虚线, 加粗, 叠在红线上方)
  const cx = col;
  depthCtx.strokeStyle = '#00ff00';
  depthCtx.lineWidth = 2;
  depthCtx.setLineDash([4, 4]);
  depthCtx.beginPath();
  depthCtx.moveTo(cx + 0.5, STRIP + 0.5);
  depthCtx.lineTo(cx + 0.5, H - 0.5);
  depthCtx.stroke();
  depthCtx.setLineDash([]);

  // 左右箭头: 表示该线可左右滑动
  drawLeftRightArrows(depthCtx, cx + 0.5, STRIP + RES / 4, '#ffffff', depthHovered ? ARROW_ALPHA_HOVER : ARROW_ALPHA_IDLE);
}

function updateTitles(row, col) {
  const d = Math.round(row * RES_ANNO / RES);
  const c = Math.round(col * RES_ANNO / RES);
  document.getElementById('titleCol').innerHTML = 'Attention Matrix';
  document.querySelector('#colScanLab .lab-r').textContent = `${c})`;
  document.getElementById('titleDepth').innerHTML = 'Spatial Mapping';
  document.querySelector('#depthScanLab .lab-r').textContent = `${d})`;
}

function updateRowLabels(row) {
  const d = Math.round(row * RES_ANNO / RES);
  // 图1/图2: 红线在 y=row, 行号写在红线左端
  const scaleO = origCanvas.offsetWidth / RES;
  const pyO = row * scaleO;
  const H = origCanvas.offsetHeight;
  const lab = document.getElementById('origRowLab');
  lab.innerHTML = 'depth<br>' + d;
  const ch = lab.offsetHeight;
  lab.style.left = (-lab.offsetWidth - 2) + 'px';
  lab.style.top = clamp(pyO - ch / 2, 0, H - ch) + 'px';
  lab.style.transform = '';
  // 图4: 红线在 y=STRIP+row (热图内), 行号写在红线左端
  const scaleDY = depthCanvas.offsetHeight / (STRIP + RES);
  const pyD = (STRIP + row) * scaleDY;
  const dLab = document.getElementById('depthRowLab');
  dLab.innerHTML = 'depth<br>' + d;
  const dch = dLab.offsetHeight;
  dLab.style.left = '2px';
  dLab.style.top = clamp(pyD - dch / 2, 0, depthCanvas.offsetHeight - dch) + 'px';
  dLab.style.transform = '';
  // 图4: 红线左侧图像外写 Q (query 深度)
  const qLab = document.getElementById('depthQLab');
  const qh = qLab.offsetHeight;
  qLab.style.left = (-qLab.offsetWidth - 2) + 'px';
  qLab.style.top = clamp(pyD - qh / 2, 0, depthCanvas.offsetHeight - qh) + 'px';
}

function updateColLabels(col) {
  const c = Math.round(col * RES_ANNO / RES);
  // 图1/图2: 绿竖线在 x=col, 列号写在图像顶端
  const scaleO = origCanvas.offsetWidth / RES;
  const pxO = col * scaleO;
  const W = origCanvas.offsetWidth;
  for (const id of ['orig', 'gt']) {
    const lab = document.getElementById(id + 'ColLab');
    lab.textContent = 'A-scan ' + c;
    const cw = lab.offsetWidth;
    let lx = pxO - cw / 2;
    lx = clamp(lx, 0, W - cw);
    lab.style.left = lx + 'px';
    lab.style.top = '-14px';
    lab.style.transform = '';
  }
  // 图4: 绿竖线在 x=col (热图内), 列号写在绿线顶端(热图顶部)
  const scaleDX = depthCanvas.offsetWidth / RES;
  const scaleDY = depthCanvas.offsetHeight / (STRIP + RES);
  const pxD = col * scaleDX;
  const dLab = document.getElementById('depthColLab');
  dLab.textContent = 'A-scan ' + c;
  const dcw = dLab.offsetWidth;
  const DW = depthCanvas.offsetWidth;
  let dx = pxD - dcw / 2;
  dx = clamp(dx, 0, DW - dcw);
  dLab.style.left = dx + 'px';
  dLab.style.top = (STRIP * scaleDY + 2) + 'px';
  dLab.style.transform = '';
}

function updateHover(col, row) {
  curCol = col;
  curRow = row;
  renderPanel(origCtx, origImg, row, col);
  renderPanel(gtCtx, gtImg, row, col);
  renderColPanel(col, row);
  renderDepthPanel(row, col);
  updateTitles(row, col);
  updateRowLabels(row);
  updateColLabels(col);
}

function pickPoint(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  const col = clamp(Math.floor((e.clientX - rect.left) / rect.width * RES), 0, RES - 1);
  const row = clamp(Math.floor((e.clientY - rect.top) / rect.height * RES), 0, RES - 1);
  return [col, row];
}

// 图4 坐标: 热图在 (0, STRIP) ~ (224, 248), x=A-scan(列), y=depth(行)
function pickPointDepth(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  const col = clamp(Math.floor((e.clientX - rect.left) / rect.width * RES), 0, RES - 1);
  const row = clamp(Math.floor((e.clientY - rect.top - STRIP) / (rect.height - STRIP) * RES), 0, RES - 1);
  return [col, row];
}

// 图1: 划过预览(锁定后显示虚点跟随鼠标)
function onHover(e) {
  const [col, row] = pickPoint(e);
  if (locked) {
    origHoverCol = col;
    origHoverRow = row;
    renderPanel(origCtx, origImg, pinnedRow, pinnedCol);
    drawOrigHoverAnchor();
    return;
  }
  origHoverCol = null;
  origHoverRow = null;
  updateHover(col, row);
}

// 图1: 单击 —— 记录并锁定(不再跟随鼠标), 同时右下角鼠标图标"单击"
function onPin(e) {
  const [col, row] = pickPoint(e);
  pinnedCol = col;
  pinnedRow = row;
  updateHover(col, row);
  locked = true;
  const icon = document.getElementById('origMouseIcon');
  icon.classList.add('clicking');
  setTimeout(() => icon.classList.remove('clicking'), 120);
}

// 图1: 重新进入 —— 解锁(恢复跟随)
function onEnter(e) {
  locked = false;
  origHoverCol = null;
  origHoverRow = null;
}

// 图1: 单击锁定后, 悬停位置画半透明虚点
function drawOrigHoverAnchor() {
  if (origHoverCol === null || origHoverRow === null) return;
  origCtx.globalAlpha = 0.45;
  origCtx.fillStyle = '#ffffff';
  origCtx.beginPath(); origCtx.arc(origHoverCol, origHoverRow, 6, 0, Math.PI * 2); origCtx.fill();
  origCtx.strokeStyle = '#000000';
  origCtx.lineWidth = 1;
  origCtx.beginPath(); origCtx.arc(origHoverCol, origHoverRow, 6, 0, Math.PI * 2); origCtx.stroke();
  origCtx.globalAlpha = 1.0;
}

// 图2: 单击 —— 仅记录(不锁定图1), 同时让右下角鼠标图标"单击"
function onPinGt(e) {
  const [col, row] = pickPoint(e);
  pinnedCol = col;
  pinnedRow = row;
  updateHover(col, row);
  const icon = document.getElementById('gtMouseIcon');
  icon.classList.add('clicking');
  setTimeout(() => icon.classList.remove('clicking'), 120);
}

// 图2: 悬停时画半透明锚点
function drawGtHoverAnchor() {
  if (gtHoverCol === null || gtHoverRow === null) return;
  gtCtx.globalAlpha = 0.45;
  gtCtx.fillStyle = '#ffffff';
  gtCtx.beginPath(); gtCtx.arc(gtHoverCol, gtHoverRow, 6, 0, Math.PI * 2); gtCtx.fill();
  gtCtx.strokeStyle = '#000000';
  gtCtx.lineWidth = 1;
  gtCtx.beginPath(); gtCtx.arc(gtHoverCol, gtHoverRow, 6, 0, Math.PI * 2); gtCtx.stroke();
  gtCtx.globalAlpha = 1.0;
}

function onGtMove(e) {
  const [col, row] = pickPoint(e);
  gtHoverCol = col;
  gtHoverRow = row;
  renderPanel(gtCtx, gtImg, pinnedRow, pinnedCol);
  drawGtHoverAnchor();
}

function onGtEnter() {
  document.getElementById('gtMouseIcon').classList.add('show');
}

function onGtLeave() {
  gtHoverCol = null;
  gtHoverRow = null;
  renderPanel(gtCtx, gtImg, pinnedRow, pinnedCol);
  document.getElementById('gtMouseIcon').classList.remove('show');
}

// 图4: 划过 —— 只左右滑动(改 A-scan/列), 不改 depth(行); 锁定后显示灰色竖线跟随
function onHoverDepth(e) {
  const [col] = pickPointDepth(e);
  if (lockedDepth) {
    depthHoverCol = col;
    renderDepthPanel(pinnedRow, pinnedCol);
    drawDepthHoverLine();
    return;
  }
  depthHoverCol = null;
  updateHover(col, pinnedRow);
}

// 图4: 单击 —— 确定 A-scan(列) 并锁定, 同时右下角鼠标图标"单击"
function onPinDepth(e) {
  const [col] = pickPointDepth(e);
  pinnedCol = col;
  updateHover(col, pinnedRow);
  lockedDepth = true;
  const icon = document.getElementById('depthMouseIcon');
  icon.classList.add('clicking');
  setTimeout(() => icon.classList.remove('clicking'), 120);
}

// 图4: 重新进入 —— 解锁(恢复左右滑动)
function onEnterDepth(e) {
  lockedDepth = false;
  depthHoverCol = null;
}

// 图4: 单击锁定后, 悬停列画灰色竖线(填充/描边同虚点)
function drawDepthHoverLine() {
  if (depthHoverCol === null) return;
  const H = STRIP + RES;
  const cx = depthHoverCol;
  depthCtx.globalAlpha = 0.45;
  depthCtx.fillStyle = '#ffffff';
  depthCtx.fillRect(cx - 2, STRIP, 4, RES);
  depthCtx.strokeStyle = '#000000';
  depthCtx.lineWidth = 1;
  depthCtx.strokeRect(cx - 2 + 0.5, STRIP + 0.5, 3, RES - 1);
  depthCtx.globalAlpha = 1.0;
}

// 图3: 划过 —— 只改查询深度(row), 不改 A-scan(列); 锁定后显示灰色横线跟随
function onMoveCol(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  const row = clamp(Math.floor((e.clientY - rect.top - STRIP) / (rect.height - STRIP) * RES), 0, RES - 1);
  if (lockedCol) {
    colHoverRow = row;
    renderColPanel(pinnedCol, pinnedRow);
    drawColHoverLine();
    return;
  }
  colHoverRow = null;
  updateHover(pinnedCol, row);
}

// 图3: 单击 —— 确定查询深度(row) 并锁定, 同时右下角鼠标图标"单击"
function onPinCol(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  const row = clamp(Math.floor((e.clientY - rect.top - STRIP) / (rect.height - STRIP) * RES), 0, RES - 1);
  pinnedRow = row;
  updateHover(pinnedCol, row);
  lockedCol = true;
  const icon = document.getElementById('colMouseIcon');
  icon.classList.add('clicking');
  setTimeout(() => icon.classList.remove('clicking'), 120);
}

// 图3: 重新进入 —— 解锁(恢复移动)
function onEnterCol(e) {
  lockedCol = false;
  colHoverRow = null;
}

// 图3: 单击锁定后, 悬停行画灰色横线(填充/描边同虚点)
function drawColHoverLine() {
  if (colHoverRow === null) return;
  const S = STRIP + RES;
  const ly = STRIP + colHoverRow + 0.5;
  const x0 = STRIP - 4;
  const x1 = S - 4;
  colCtx.globalAlpha = 0.45;
  colCtx.fillStyle = '#ffffff';
  colCtx.fillRect(x0, ly - 2, x1 - x0, 4);
  colCtx.strokeStyle = '#000000';
  colCtx.lineWidth = 1;
  colCtx.strokeRect(x0 + 0.5, ly - 2 + 0.5, x1 - x0 - 1, 3);
  colCtx.globalAlpha = 1.0;
}

// 移出图像: 回到最后一次单击的位置
function onLeave(e) {
  origHoverCol = null;
  origHoverRow = null;
  colHoverRow = null;
  depthHoverCol = null;
  updateHover(pinnedCol, pinnedRow);
}

function drawColorbar(ctx) {
  const w = 16, h = 224;
  ctx.clearRect(0, 0, w, h);
  for (let i = 0; i < h; i++) {
    const v = Math.round(255 * (h - 1 - i) / (h - 1));   // 上=max(黄), 下=min(紫)
    ctx.fillStyle = `rgb(${virLUT[v * 4]},${virLUT[v * 4 + 1]},${virLUT[v * 4 + 2]})`;
    ctx.fillRect(0, i, 16, 1);
  }
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, 15, h - 1);
}

const HCMS_LEGEND = [
  { color: [170, 170, 170], name: 'BG (top)' },
  { color: [173, 216, 230], name: 'RNFL' },
  { color: [0, 255, 255], name: 'GCIPL' },
  { color: [0, 128, 0], name: 'INL' },
  { color: [255, 255, 0], name: 'OPL' },
  { color: [255, 165, 0], name: 'ONL' },
  { color: [255, 0, 0], name: 'IS' },
  { color: [139, 0, 0], name: 'OS' },
  { color: [0, 255, 0], name: 'RPE' },
  { color: [60, 60, 60], name: 'BG (bottom)' },
];

function buildLegend(legend, boxId = 'legend') {
  const box = document.getElementById(boxId);
  box.innerHTML = '';
  for (const it of legend) {
    const item = document.createElement('div');
    item.className = 'item';
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = `rgb(${it.color[0]},${it.color[1]},${it.color[2]})`;
    const tx = document.createElement('span');
    tx.textContent = it.name;
    item.appendChild(sw);
    item.appendChild(tx);
    box.appendChild(item);
  }
}

function legendFor(key) {
  if (key === 'nAMD') return LEGEND;
  if (key === 'dme') return DUKE_LEGEND;
  return HCMS_LEGEND;
}

// 水平(列间)注意力小节: 切换数据集(dme/nAMD/normal)
let hDatasetReq = 0;
async function setHDataset(key) {
  const req = ++hDatasetReq;
  hCurDataset = key;
  const d = await ensureHDataset(key);
  if (req !== hDatasetReq) return;   // 期间有更新的切换请求, 放弃本次渲染
  hOrigImg = d.origImg; hGtImg = d.gtImg; hAttnU8 = d.attnU8;
  hAttnColorCache = null;
  hGtStripH = computeGtStrip(d.gtImg);
  hGtStripV = transposeStrip(hGtStripH);
  document.getElementById('hThumbD').classList.toggle('active', key === 'dme');
  document.getElementById('hThumbN').classList.toggle('active', key === 'nAMD');
  document.getElementById('hThumbH').classList.toggle('active', key === 'normal');
  buildLegend(legendFor(key), 'hLegend');
  renderHCol();
}

// 加粗红色线 (无描边)
function strokeRedBold(ctx, x1, y1, x2, y2) {
  ctx.strokeStyle = '#ff0000';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

// 灰色竖线(锁定后跟随鼠标, 白底黑边同虚点)
function drawGrayVBar(ctx, x) {
  ctx.globalAlpha = 0.45;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(x - 2, 0, 4, RES);
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 1;
  ctx.strokeRect(x - 2 + 0.5, 0.5, 3, RES - 1);
  ctx.globalAlpha = 1.0;
}

// 灰色横线(锁定后跟随鼠标, 白底黑边同虚点, 用于图3)
function drawGrayHBar(ctx, y) {
  ctx.globalAlpha = 0.45;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(HSTRIP, y - 2, RES, 4);
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 1;
  ctx.strokeRect(HSTRIP + 0.5, y - 2 + 0.5, RES - 1, 3);
  ctx.globalAlpha = 1.0;
}

// 重绘: 原图(红线=选中列) + 注意力矩阵(红线=query列) + 空间映射叠加
function renderHCol() {
  const c = hCol;
  // 面板1: 原图 + 红线(选中列, 加粗描边)
  hcmsOrigCtx.clearRect(0, 0, RES, RES);
  hcmsOrigCtx.drawImage(hOrigImg, 0, 0);
  strokeRedBold(hcmsOrigCtx, c + 0.5, 0.5, c + 0.5, RES - 0.5);
  if (hLocked && hHoverCol !== null && hHoverFig === 'orig') drawGrayVBar(hcmsOrigCtx, hHoverCol);
  drawLeftRightArrows(hcmsOrigCtx, c + 0.5, RES / 4, '#ffffff', hOrigHovered ? ARROW_ALPHA_HOVER : ARROW_ALPHA_IDLE);

  // 面板2: GT + 红线(选中列, 加粗描边)
  hcmsGtCtx.clearRect(0, 0, RES, RES);
  hcmsGtCtx.drawImage(hGtImg, 0, 0);
  strokeRedBold(hcmsGtCtx, c + 0.5, 0.5, c + 0.5, RES - 0.5);
  if (hHoverCol !== null && hHoverFig === 'gt') drawGrayVBar(hcmsGtCtx, hHoverCol);
  drawLeftRightArrows(hcmsGtCtx, c + 0.5, RES / 4, '#ffffff', hGtHovered ? ARROW_ALPHA_HOVER : ARROW_ALPHA_IDLE);

  // 图1/图2: 红色竖线上端图像外侧写 A-scan n (字体字号同 IVSA 图1)
  const cAnno = Math.round(c * RES_ANNO / RES);
  const scaleHX = hcmsOrigCanvas.offsetWidth / RES;
  const pxH = c * scaleHX;
  for (const id of ['hcmsOrig', 'hcmsGt']) {
    const lab = document.getElementById(id + 'ColLab');
    lab.textContent = 'A-scan ' + cAnno;
    const cw = lab.offsetWidth;
    let lx = pxH - cw / 2;
    lx = clamp(lx, 0, hcmsOrigCanvas.offsetWidth - cw);
    lab.style.left = lx + 'px';
    lab.style.top = '-10px';
  }

  // 图4: 红线上方图像外写 Q(A-scan n) (跟随红线, 与 Key 水平对齐, 括号固定宽度)
  const qLab = document.getElementById('hSpatialQLab');
  qLab.querySelector('.lab-r').textContent = cAnno + ')';
  const qw = qLab.offsetWidth;
  let qx = pxH - qw / 2;
  qx = clamp(qx, 0, hSpatialCanvas.offsetWidth - qw);
  qLab.style.left = qx + 'px';
  qLab.style.top = '-10px';

  // 面板3: 注意力矩阵(偏移 HSTRIP) + GT 色条(顶/左) + 红线(query 列)
  if (!hAttnColorCache) hAttnColorCache = colorize(hAttnU8, 0);
  hAttnCtx.clearRect(0, 0, HSTRIP + RES, HSTRIP + RES);
  hAttnCtx.putImageData(hAttnColorCache, HSTRIP, HSTRIP);
  if (hGtStripH) hAttnCtx.drawImage(hGtStripH, HSTRIP, 0);       // 顶部: Key 轴
  if (hGtStripV) hAttnCtx.drawImage(hGtStripV, 0, HSTRIP);       // 左侧: Query 轴
  strokeRedBold(hAttnCtx, HSTRIP + 0.5, HSTRIP + c + 0.5, HSTRIP + RES - 0.5, HSTRIP + c + 0.5);
  if (hLocked && hHoverCol !== null && hHoverFig === 'attn') drawGrayHBar(hAttnCtx, HSTRIP + hHoverCol);

  // 上下箭头: 表示该线可上下滑动
  drawUpDownArrows(hAttnCtx, HSTRIP + RES - 9, HSTRIP + c + 0.5, '#ffffff', hAttnHovered ? ARROW_ALPHA_HOVER : ARROW_ALPHA_IDLE);

  // 面板4: 空间映射 = 列 c 的注意力沿深度复制, 原图30% + 热图70%
  const row = hAttnU8.subarray(c * RES, (c + 1) * RES);
  hSpatialCtx.putImageData(colorizeHeat(row), 0, 0);
  hSpatialCtx.globalAlpha = 0.30;
  hSpatialCtx.drawImage(hOrigImg, 0, 0);
  hSpatialCtx.globalAlpha = 1.0;
  strokeRedBold(hSpatialCtx, c + 0.5, 0.5, c + 0.5, RES - 0.5);
  if (hLocked && hHoverCol !== null && hHoverFig === 'spatial') drawGrayVBar(hSpatialCtx, hHoverCol);
  drawLeftRightArrows(hSpatialCtx, c + 0.5, RES / 4, '#ffffff', hSpatialHovered ? ARROW_ALPHA_HOVER : ARROW_ALPHA_IDLE);
}

// 面板1 交互: 划过预览 / 单击固定 / 移出回退
let hColRenderQueued = false;
function renderHColSoon() {
  if (hColRenderQueued) return;
  hColRenderQueued = true;
  requestAnimationFrame(() => {
    hColRenderQueued = false;
    renderHCol();
  });
}
function mouseIconClick(canvas) {
  const icon = canvas.parentElement.querySelector('.gt-mouse-icon');
  if (icon) {
    icon.classList.add('clicking');
    setTimeout(() => icon.classList.remove('clicking'), 120);
  }
}
function onHCol(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  const x = clamp(Math.floor((e.clientX - rect.left) / rect.width * RES), 0, RES - 1);
  hHoverFig = (e.currentTarget.id === 'hcmsOrigCanvas') ? 'orig' : 'spatial';
  if (hLocked) { hHoverCol = x; } else { hCol = x; }
  renderHColSoon();
}
function onHColPin(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  hCol = clamp(Math.floor((e.clientX - rect.left) / rect.width * RES), 0, RES - 1);
  hFixedCol = hCol;
  hLocked = true;
  hHoverCol = null;
  renderHCol();
  mouseIconClick(e.currentTarget);
}
function onHColLeave(e) {
  hHoverCol = null;
  hHoverFig = null;
  hCol = hFixedCol;
  renderHCol();
}
function onHColEnter(e) {
  hLocked = false;
  hHoverCol = null;
  hHoverFig = null;
  renderHCol();
}
// 图2(GT): 鼠标滑过时只有虚线跟随(红线仅在单击时响应, 同 IVSA 图2)
function onHColGtMove(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  const x = clamp(Math.floor((e.clientX - rect.left) / rect.width * RES), 0, RES - 1);
  hHoverFig = 'gt';
  hHoverCol = x;
  renderHColSoon();
}
// 图3(注意力矩阵): 竖轴是 query 列, 鼠标上下移动改变列
function onHAttnMove(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  const scale = rect.height / (HSTRIP + RES);
  const cy = (e.clientY - rect.top) / scale;
  const y = clamp(Math.floor(cy - HSTRIP), 0, RES - 1);
  hHoverFig = 'attn';
  if (hLocked) { hHoverCol = y; } else { hCol = y; }
  renderHColSoon();
}
function onHAttnPin(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  const scale = rect.height / (HSTRIP + RES);
  const cy = (e.clientY - rect.top) / scale;
  hCol = clamp(Math.floor(cy - HSTRIP), 0, RES - 1);
  hFixedCol = hCol;
  hLocked = true;
  hHoverCol = null;
  renderHCol();
  mouseIconClick(e.currentTarget);
}

let datasetReq = 0;
async function setDataset(key) {
  const req = ++datasetReq;
  curDataset = key;
  const d = await ensureDataset(key);
  if (req !== datasetReq) return;   // 期间有更新的切换请求, 放弃本次渲染
  origImg = d.origImg; gtImg = d.gtImg; rowMaps = d.rowMaps; colMaps = d.colMaps; gtData = d.gtData;
  document.getElementById('thumbD').classList.toggle('active', key === 'dme');
  document.getElementById('thumbN').classList.toggle('active', key === 'nAMD');
  document.getElementById('thumbH').classList.toggle('active', key === 'normal');
  buildLegend(legendFor(key));
  updateHover(pinnedCol, pinnedRow);
}

// 模型名称 / 年份 / 参考文献编号 (全局, 每个模型固定)
const qualModelNames = {
  unet: 'U-Net', deeplabv3p: 'DeepLabv3+', unet3p: 'UNet3+',
  transunet: 'TransUNet', cswinunet: 'CSWin-Unet', agileformer: 'AgileFormer',
  mambaunet: 'Mamba-UNet', swinumamba: 'Swin-UMamba', switchumamba: 'Switch-UMamba',
  relaynet: 'ReLayNet', tcct: 'TCCT-BP', gdnet: 'GD-Net',
};
const qualModelYears = {
  unet: '2015', deeplabv3p: '2018', unet3p: '2020',
  transunet: '2021', cswinunet: '2025', agileformer: '2026',
  mambaunet: '2024', swinumamba: '2024', switchumamba: '2026',
  relaynet: '2017', tcct: '2024', gdnet: '2024',
};
// 参考文献编号 (与底部 References 列表顺序一致)
const qualModelRefs = {
  unet: 1, deeplabv3p: 2, unet3p: 3,
  transunet: 4, cswinunet: 5, agileformer: 6,
  mambaunet: 7, swinumamba: 8, switchumamba: 9,
  relaynet: 10, tcct: 11, gdnet: 12,
};

// Duke / AROI: 15 列 (image, gt, 12 个模型, ours)
const QUAL_COL_FULL = {
  image: 0, gt: 1, unet: 2, deeplabv3p: 3, unet3p: 4, transunet: 5, cswinunet: 6,
  agileformer: 7, mambaunet: 8, swinumamba: 9, switchumamba: 10, relaynet: 11,
  tcct: 12, gdnet: 13, ours: 14,
};
const QUAL_SLUGS_FULL = {
  cnn: ['unet', 'deeplabv3p', 'unet3p'],
  trans: ['transunet', 'cswinunet', 'agileformer'],
  mamba: ['mambaunet', 'swinumamba', 'switchumamba'],
  oct: ['relaynet', 'tcct', 'gdnet'],
};

// HCMS: 14 列 (缺 GD-Net)
const QUAL_COL_HCMS = {
  image: 0, gt: 1, unet: 2, deeplabv3p: 3, unet3p: 4, transunet: 5, cswinunet: 6,
  agileformer: 7, mambaunet: 8, swinumamba: 9, switchumamba: 10, relaynet: 11, tcct: 12, ours: 13,
};
const QUAL_SLUGS_HCMS = {
  cnn: ['unet', 'deeplabv3p', 'unet3p'],
  trans: ['transunet', 'cswinunet', 'agileformer'],
  mamba: ['mambaunet', 'swinumamba', 'switchumamba'],
  oct: ['relaynet', 'tcct'],
};

// 每个数据集的渲染配置
const QUAL_CONFIG = {
  duke: { rows: 3, listId: 'dukeModelList', colIndex: QUAL_COL_FULL, slugs: QUAL_SLUGS_FULL },
  aroi: { rows: 2, listId: 'aroiModelList', colIndex: QUAL_COL_FULL, slugs: QUAL_SLUGS_FULL },
  hcms: { rows: 2, listId: 'hcmsModelList', colIndex: QUAL_COL_HCMS, slugs: QUAL_SLUGS_HCMS },
};

function renderQualList(ds, category) {
  const cfg = QUAL_CONFIG[ds];
  if (!cfg) return;
  const listContainer = document.getElementById(cfg.listId);
  if (!listContainer) return;
  listContainer.innerHTML = '';
  const inner = document.createElement('div');
  inner.className = 'qual-scroll-inner';
  const slugs = cfg.slugs[category] || [];
  for (let r = 1; r <= cfg.rows; r++) {
    const row = document.createElement('div');
    row.className = 'qual-scroll-row';
    slugs.forEach(slug => {
      const box = document.createElement('div');
      box.className = 'qual-box';
      const img = document.createElement('div');
      img.className = 'qual-img';
      img.style.backgroundPosition = `-${cfg.colIndex[slug] * 64}px -${(r - 1) * 64}px`;
      box.appendChild(img);
      if (r === cfg.rows) {
        const name = document.createElement('div');
        name.className = 'qual-name';
        name.textContent = qualModelNames[slug];
        const sup = document.createElement('sup');
        sup.textContent = `[${qualModelRefs[slug]}]`;
        name.appendChild(sup);
        box.appendChild(name);
        const year = document.createElement('div');
        year.className = 'qual-year';
        year.textContent = qualModelYears[slug];
        box.appendChild(year);
      }
      row.appendChild(box);
    });
    inner.appendChild(row);
  }
  listContainer.appendChild(inner);
  listContainer.scrollLeft = 0;
}

function switchModelTab(category, event) {
  const target = event && event.target;
  const wrapper = (target && target.closest)
    ? target.closest('.qual-wrapper')
    : document.querySelector('.qual-wrapper[data-ds="duke"]');
  if (!wrapper) return;
  const ds = wrapper.dataset.ds || 'duke';
  const buttons = wrapper.querySelectorAll('.qual-tab-btn');
  buttons.forEach(btn => btn.classList.remove('active'));
  if (target) target.classList.add('active');
  else if (buttons.length) buttons[0].classList.add('active');
  renderQualList(ds, category);
}

// —— 懒加载: 各数据集的加载器 + Promise 缓存 ——
const VERTICAL_LOADERS = {
  dme: async () => {
    const [o, g, rm, cm] = await Promise.all([
      loadImage('dme_orig.png'), loadImage('dme_gt.png'),
      loadBinGz('dme_row_maps.bin'), loadBinGz('dme_col_maps.bin'),
    ]);
    return { origImg: o, gtImg: g, rowMaps: rm, colMaps: cm, gtData: getImgData(g, RES, RES).data };
  },
  nAMD: async () => {
    const [o, g, rm, cm] = await Promise.all([
      loadImage('orig.png'), loadImage('gt.png'),
      loadBinGz('row_maps.bin'), loadBinGz('col_maps.bin'),
    ]);
    return { origImg: o, gtImg: g, rowMaps: rm, colMaps: cm, gtData: getImgData(g, RES, RES).data };
  },
  normal: async () => {
    const [o, g, rm, cm] = await Promise.all([
      loadImage('hcms_orig.png'), loadImage('hcms_gt.png'),
      loadBinGz('hcms_row_maps.bin'), loadBinGz('hcms_col_maps.bin'),
    ]);
    return { origImg: o, gtImg: g, rowMaps: rm, colMaps: cm, gtData: getImgData(g, RES, RES).data };
  },
};

const HORIZONTAL_LOADERS = {
  dme: async () => {
    const [o, g, a] = await Promise.all([
      loadImage('dme_orig.png'), loadImage('dme_gt.png'), loadBin('dme_bscan_attn.bin'),
    ]);
    return { origImg: o, gtImg: g, attnU8: a };
  },
  nAMD: async () => {
    const [o, g, a] = await Promise.all([
      loadImage('orig.png'), loadImage('gt.png'), loadBin('aroi_bscan_attn.bin'),
    ]);
    return { origImg: o, gtImg: g, attnU8: a };
  },
  normal: async () => {
    const [o, g, a] = await Promise.all([
      loadImage('hcms_orig.png'), loadImage('hcms_gt.png'), loadBin('bscan_attn.bin'),
    ]);
    return { origImg: o, gtImg: g, attnU8: a };
  },
};

const DATASET_PROMISES = {};
function ensureDataset(key) {
  if (DATASETS[key]) return Promise.resolve(DATASETS[key]);
  if (!DATASET_PROMISES[key]) {
    DATASET_PROMISES[key] = VERTICAL_LOADERS[key]().then(d => { DATASETS[key] = d; return d; });
  }
  return DATASET_PROMISES[key];
}

const HDATASET_PROMISES = {};
function ensureHDataset(key) {
  if (HDATASETS[key]) return Promise.resolve(HDATASETS[key]);
  if (!HDATASET_PROMISES[key]) {
    HDATASET_PROMISES[key] = HORIZONTAL_LOADERS[key]().then(d => { HDATASETS[key] = d; return d; });
  }
  return HDATASET_PROMISES[key];
}

// —— 自动动: 加载后锚点自动扫动, 鼠标首次介入即停止(此后不再自动动) ——
let autoAnimStopped = false;
let autoAnimRAF = null;

const AUTO_DATASETS = ['dme', 'nAMD', 'normal'];
let autoDatasetIdx = 0;      // 当前显示的样本 (dme)
let autoSwitching = false;

// 换到下一个样本 (不重置锚点位置, 动画继续)
async function autoAdvanceDataset() {
  if (autoSwitching || autoAnimStopped) return;
  autoSwitching = true;
  try {
    autoDatasetIdx = (autoDatasetIdx + 1) % AUTO_DATASETS.length;
    const key = AUTO_DATASETS[autoDatasetIdx];
    const d = await ensureDataset(key);
    if (autoAnimStopped) return;
    origImg = d.origImg; gtImg = d.gtImg; rowMaps = d.rowMaps; colMaps = d.colMaps; gtData = d.gtData;
    curDataset = key;
    document.getElementById('thumbD').classList.toggle('active', key === 'dme');
    document.getElementById('thumbN').classList.toggle('active', key === 'nAMD');
    document.getElementById('thumbH').classList.toggle('active', key === 'normal');
    buildLegend(legendFor(key));
  } catch (e) {
    console.warn('auto advance failed:', e.message);
  } finally {
    autoSwitching = false;
  }
}

function stopAutoAnim() {
  if (autoAnimStopped) return;
  autoAnimStopped = true;
  if (autoAnimRAF) { cancelAnimationFrame(autoAnimRAF); autoAnimRAF = null; }
}

function startAutoAnim() {
  if (autoAnimStopped) return;
  const cx = DEFAULT_COL;   // 正方形中心列
  const cy = DEFAULT_ROW;   // 正方形中心行
  const s = 50;             // 半边长 (小正方形)
  const perim = 8 * s;      // 周长
  const half = perim / 2;   // 半个周长
  let p = 0;                // 累计沿周长位置
  let halfCount = 0;        // 已走过的半周长段数
  const step = () => {
    if (autoAnimStopped) return;
    p += 1;
    const hc = Math.floor(p / half);
    if (hc !== halfCount) {
      halfCount = hc;
      autoAdvanceDataset();   // 每走半个周长, 若未介入则换下一个样本
    }
    const r = p % perim;
    let col, row;
    if (r < 2 * s) {                 // 上边: 向右
      col = cx - s + r; row = cy - s;
    } else if (r < 4 * s) {          // 右边: 向下
      col = cx + s; row = cy - s + (r - 2 * s);
    } else if (r < 6 * s) {          // 下边: 向左
      col = cx + s - (r - 4 * s); row = cy + s;
    } else {                          // 左边: 向上
      col = cx - s; row = cy + s - (r - 6 * s);
    }
    updateHover(clamp(Math.round(col), 0, RES - 1), clamp(Math.round(row), 0, RES - 1));
    autoAnimRAF = requestAnimationFrame(step);
  };
  autoAnimRAF = requestAnimationFrame(step);
}

// —— IHSA 自动动: 加载后选中列左右扫动, 鼠标首次介入即停止 ——
let hAutoAnimStopped = false;
let hAutoAnimRAF = null;

const H_AUTO_DATASETS = ['dme', 'nAMD', 'normal'];
let hAutoDatasetIdx = 0;      // 当前显示的样本 (dme)
let hAutoSwitching = false;

// 换到下一个样本 (水平小节, 不重置选中列)
async function autoAdvanceHDataset() {
  if (hAutoSwitching || hAutoAnimStopped) return;
  hAutoSwitching = true;
  try {
    hAutoDatasetIdx = (hAutoDatasetIdx + 1) % H_AUTO_DATASETS.length;
    const key = H_AUTO_DATASETS[hAutoDatasetIdx];
    const d = await ensureHDataset(key);
    if (hAutoAnimStopped) return;
    hOrigImg = d.origImg; hGtImg = d.gtImg; hAttnU8 = d.attnU8;
    hCurDataset = key;
    hAttnColorCache = null;
    hGtStripH = computeGtStrip(d.gtImg);
    hGtStripV = transposeStrip(hGtStripH);
    document.getElementById('hThumbD').classList.toggle('active', key === 'dme');
    document.getElementById('hThumbN').classList.toggle('active', key === 'nAMD');
    document.getElementById('hThumbH').classList.toggle('active', key === 'normal');
    buildLegend(legendFor(key), 'hLegend');
  } catch (e) {
    console.warn('auto advance (horizontal) failed:', e.message);
  } finally {
    hAutoSwitching = false;
  }
}

function stopHAutoAnim() {
  if (hAutoAnimStopped) return;
  hAutoAnimStopped = true;
  if (hAutoAnimRAF) { cancelAnimationFrame(hAutoAnimRAF); hAutoAnimRAF = null; }
}

function startHAutoAnim() {
  if (hAutoAnimStopped) return;
  const sweepHalf = 100;                   // 扫动半范围 100px (总范围 200px)
  const center = Math.round(RES / 2);      // 中心列 (112)
  const left = center - sweepHalf;          // 左端
  const right = center + sweepHalf;         // 右端
  let col = left;                           // 从左开始
  let dir = 1;                              // 1=向右, -1=向左
  const step = () => {
    if (hAutoAnimStopped) return;
    hCol = col;
    renderHCol();
    col += dir;
    if (col > right) {                      // 到右端: 换样本并掉头向左
      col = right;
      dir = -1;
      autoAdvanceHDataset();
    } else if (col < left) {                // 到左端: 换样本并掉头向右
      col = left;
      dir = 1;
      autoAdvanceHDataset();
    }
    hAutoAnimRAF = requestAnimationFrame(step);
  };
  hAutoAnimRAF = requestAnimationFrame(step);
}

async function init() {
  drawColorbar(cbCtx);
  drawColorbar(hCbCtx);

  // 懒加载: 只先加载默认数据集 (dme)
  await setDataset('dme');

  // 自动动: 锚点自动扫动, 每半个周长切换样本; 鼠标首次进入即停止
  startAutoAnim();
  ensureDataset('nAMD').catch(() => {});
  ensureDataset('normal').catch(() => {});
  [origCanvas, gtCanvas, colCanvas, depthCanvas,
   document.getElementById('thumbD'), document.getElementById('thumbN'), document.getElementById('thumbH')
  ].forEach(el => el.addEventListener('mouseenter', stopAutoAnim));

  document.getElementById('thumbD').addEventListener('click', () => setDataset('dme'));
  document.getElementById('thumbN').addEventListener('click', () => setDataset('nAMD'));
  document.getElementById('thumbH').addEventListener('click', () => setDataset('normal'));
  document.getElementById('thumbD').addEventListener('mouseenter', () => setDataset('dme'));
  document.getElementById('thumbN').addEventListener('mouseenter', () => setDataset('nAMD'));
  document.getElementById('thumbH').addEventListener('mouseenter', () => setDataset('normal'));

  try {
    // 懒加载: 只先加载默认数据集 (dme)
    await setHDataset('dme');

    // 自动动: 选中列左右扫动, 鼠标首次进入即停止
    startHAutoAnim();
    [hcmsOrigCanvas, hcmsGtCanvas, hAttnCanvas, hSpatialCanvas].forEach(c => c.addEventListener('mouseenter', stopHAutoAnim));

    hcmsOrigCanvas.addEventListener('mousemove', onHCol);
    hcmsOrigCanvas.addEventListener('click', onHColPin);
    hcmsOrigCanvas.addEventListener('mouseleave', onHColLeave);
    hcmsOrigCanvas.addEventListener('mouseenter', onHColEnter);
    hcmsOrigCanvas.addEventListener('mouseenter', () => { hOrigHovered = true; renderHCol(); });
    hcmsOrigCanvas.addEventListener('mouseleave', () => { hOrigHovered = false; renderHCol(); });
    hcmsOrigCanvas.addEventListener('mouseenter', () => { document.getElementById('hcmsOrigMouseIcon').classList.add('show'); });
    hcmsOrigCanvas.addEventListener('mouseleave', () => { document.getElementById('hcmsOrigMouseIcon').classList.remove('show'); });

    hcmsGtCanvas.addEventListener('mousemove', onHColGtMove);
    hcmsGtCanvas.addEventListener('click', onHColPin);
    hcmsGtCanvas.addEventListener('mouseleave', onHColLeave);
    hcmsGtCanvas.addEventListener('mouseenter', onHColEnter);
    hcmsGtCanvas.addEventListener('mouseenter', () => { hGtHovered = true; renderHCol(); });
    hcmsGtCanvas.addEventListener('mouseleave', () => { hGtHovered = false; renderHCol(); });
    hcmsGtCanvas.addEventListener('mouseenter', () => { document.getElementById('hcmsGtMouseIcon').classList.add('show'); });
    hcmsGtCanvas.addEventListener('mouseleave', () => { document.getElementById('hcmsGtMouseIcon').classList.remove('show'); });

    hAttnCanvas.addEventListener('mousemove', onHAttnMove);
    hAttnCanvas.addEventListener('click', onHAttnPin);
    hAttnCanvas.addEventListener('mouseleave', onHColLeave);
    hAttnCanvas.addEventListener('mouseenter', onHColEnter);
    hAttnCanvas.addEventListener('mouseenter', () => { hAttnHovered = true; renderHCol(); });
    hAttnCanvas.addEventListener('mouseleave', () => { hAttnHovered = false; renderHCol(); });
    hAttnCanvas.addEventListener('mouseenter', () => { document.getElementById('hAttnMouseIcon').classList.add('show'); });
    hAttnCanvas.addEventListener('mouseleave', () => { document.getElementById('hAttnMouseIcon').classList.remove('show'); });

    hSpatialCanvas.addEventListener('mousemove', onHCol);
    hSpatialCanvas.addEventListener('click', onHColPin);
    hSpatialCanvas.addEventListener('mouseleave', onHColLeave);
    hSpatialCanvas.addEventListener('mouseenter', onHColEnter);
    hSpatialCanvas.addEventListener('mouseenter', () => { hSpatialHovered = true; renderHCol(); });
    hSpatialCanvas.addEventListener('mouseleave', () => { hSpatialHovered = false; renderHCol(); });
    hSpatialCanvas.addEventListener('mouseenter', () => { document.getElementById('hSpatialMouseIcon').classList.add('show'); });
    hSpatialCanvas.addEventListener('mouseleave', () => { document.getElementById('hSpatialMouseIcon').classList.remove('show'); });
    document.getElementById('hThumbD').addEventListener('click', () => setHDataset('dme'));
    document.getElementById('hThumbN').addEventListener('click', () => setHDataset('nAMD'));
    document.getElementById('hThumbH').addEventListener('click', () => setHDataset('normal'));
    document.getElementById('hThumbD').addEventListener('mouseenter', () => setHDataset('dme'));
    document.getElementById('hThumbN').addEventListener('mouseenter', () => setHDataset('nAMD'));
    document.getElementById('hThumbH').addEventListener('mouseenter', () => setHDataset('normal'));
  } catch (e) {
    console.warn('horizontal section skipped:', e.message);
  }

  origCanvas.addEventListener('mousemove', onHover);
  origCanvas.addEventListener('click', onPin);
  origCanvas.addEventListener('mouseleave', onLeave);
  origCanvas.addEventListener('mouseenter', onEnter);
  origCanvas.addEventListener('mouseenter', () => { document.getElementById('origMouseIcon').classList.add('show'); });
  origCanvas.addEventListener('mouseleave', () => { document.getElementById('origMouseIcon').classList.remove('show'); });
  gtCanvas.addEventListener('click', onPinGt);
  gtCanvas.addEventListener('mousemove', onGtMove);
  gtCanvas.addEventListener('mouseenter', onGtEnter);
  gtCanvas.addEventListener('mouseleave', onGtLeave);
  depthCanvas.addEventListener('mousemove', onHoverDepth);
  depthCanvas.addEventListener('click', onPinDepth);
  depthCanvas.addEventListener('mouseleave', onLeave);
  depthCanvas.addEventListener('mouseenter', onEnterDepth);
  depthCanvas.addEventListener('mouseenter', () => { depthHovered = true; renderDepthPanel(curRow, curCol); });
  depthCanvas.addEventListener('mouseleave', () => { depthHovered = false; renderDepthPanel(curRow, curCol); });
  depthCanvas.addEventListener('mouseenter', () => { document.getElementById('depthMouseIcon').classList.add('show'); });
  depthCanvas.addEventListener('mouseleave', () => { document.getElementById('depthMouseIcon').classList.remove('show'); });
  colCanvas.addEventListener('mousemove', onMoveCol);
  colCanvas.addEventListener('click', onPinCol);
  colCanvas.addEventListener('mouseleave', onLeave);
  colCanvas.addEventListener('mouseenter', onEnterCol);
  colCanvas.addEventListener('mouseenter', () => { colHovered = true; renderColPanel(curCol, curRow); });
  colCanvas.addEventListener('mouseleave', () => { colHovered = false; renderColPanel(curCol, curRow); });
  colCanvas.addEventListener('mouseenter', () => { document.getElementById('colMouseIcon').classList.add('show'); });
  colCanvas.addEventListener('mouseleave', () => { document.getElementById('colMouseIcon').classList.remove('show'); });

  switchModelTab('cnn');
  switchModelTab('cnn', { target: document.querySelector('.qual-wrapper[data-ds="aroi"] .qual-tab-btn') });
  switchModelTab('cnn', { target: document.querySelector('.qual-wrapper[data-ds="hcms"] .qual-tab-btn') });
}

init().catch(err => {
  console.error('init failed:', err.message,
    '(serve this folder with `python -m http.server` and open the page)');
});
