# 📊 How to View Interactive Visualizations

Since GitHub shows raw code instead of running HTML files, here are simple ways to view the interactive visualizations locally.

## 🚀 Quick Methods

### Method 1: Right-Click Download (Most Reliable) ⭐
1. **Navigate to the graphics folder**: Go to https://github.com/bhoult/correlation/tree/main/docs/graphics
2. **Click on each file** to view it, then right-click "Raw" button and select "Save link as...":
   - `correlation_visualizer.html` (main interactive demo)
   - `correlation_visualizer.js` (required JavaScript)
   - `generate_static_graphics.html` (static graphics tool)

3. **Save all files to the same folder** on your computer

4. **Open in browser**: Double-click `correlation_visualizer.html` or `generate_static_graphics.html`

> ✅ **This method works 100% of the time and ensures you get the latest version of all files.**

### Method 2: Clone Repository
```bash
# Clone the repository
git clone https://github.com/bhoult/correlation.git

# Navigate to graphics folder
cd correlation/docs/graphics

# Open files in your browser
open correlation_visualizer.html          # macOS
start correlation_visualizer.html         # Windows
xdg-open correlation_visualizer.html      # Linux
```

### Method 3: GitHub Download Links (One-Click Download)
Use these direct download links to save files to your computer:

**Interactive Visualizer Files:**
- [📊 Download correlation_visualizer.html](https://github.com/bhoult/correlation/raw/main/docs/graphics/correlation_visualizer.html)
- [📜 Download correlation_visualizer.js](https://github.com/bhoult/correlation/raw/main/docs/graphics/correlation_visualizer.js)

**Static Graphics Generator:**
- [🖼️ Download generate_static_graphics.html](https://github.com/bhoult/correlation/raw/main/docs/graphics/generate_static_graphics.html)

**Instructions:**
1. Click each download link above
2. Save all files to the same folder on your computer
3. Double-click the HTML files to open in your browser

> ⚠️ **Important**: You need both the HTML and JS files for the interactive visualizer to work properly.

## 🎯 What You'll Get

### Interactive Correlation Visualizer
- **Drag data points** to see correlation change in real-time
- **Adjust time windows** and offsets interactively
- **Explore statistical significance** with Cohen's guidelines
- **Watch animated workflow** of data processing steps
- **Visualize parallel processing** across CPU cores

### Static Graphics Generator
- **Generate publication-ready diagrams** for academic papers
- **Export PNG images** for presentations
- **Professional mathematical visualizations**
- **Technical architecture diagrams**

## 🔧 Technical Requirements

### Minimum Requirements
- ✅ Modern web browser (Chrome, Firefox, Safari, Edge)
- ✅ JavaScript enabled
- ✅ HTML5 Canvas support

### File Dependencies
The visualizations require these files to be in the same directory:
- `correlation_visualizer.html` (main interface)
- `correlation_visualizer.js` (JavaScript functionality)
- `generate_static_graphics.html` (static graphics tool)

## 🎨 Features Demo

### Interactive Correlation Demo
1. Open `correlation_visualizer.html`
2. Try dragging the red data points around
3. Watch the correlation coefficient update in real-time
4. See significance thresholds change dynamically

### Time Window Analysis
1. Use the sliders to adjust window sizes (1-30 days)
2. Modify the offset (-14 to +14 days)
3. Observe how parameters affect correlation analysis
4. Read the interpretation text updates

### Statistical Significance Explorer
1. Adjust correlation strength with the slider
2. Change sample size to see impact
3. Watch Cohen's effect size classification update
4. See when correlations become "significant"

### Workflow Animation
1. Click "Start Animation" 
2. Watch each processing step highlight
3. Adjust animation speed with slider
4. See progress tracking in real-time

### Parallel Processing Visualization
1. Select number of CPU cores
2. Adjust task count
3. Click "Start Processing"
4. Watch tasks distribute across cores

## 📖 Educational Value

These visualizations help understand:

### Mathematical Concepts
- How individual data points contribute to correlation
- Visual representation of variance explanation (R²)
- Cohen's effect size guidelines with practical examples

### Technical Implementation  
- Impact of different time window sizes
- Lead-lag relationships in financial data
- How computational workload distributes across CPU cores

### Statistical Significance
- Why certain correlation values are considered significant
- Impact of sample size on statistical validity
- Difference between practical and statistical significance

## 🔄 Alternative: Online Hosting

If you want to share these visualizations online without GitHub Pages:

### Free Options
- **CodePen**: Upload HTML/JS and get shareable link
- **JSFiddle**: Create interactive demos online  
- **Netlify Drop**: Drag and drop files for instant hosting
- **Surge.sh**: Simple command-line hosting

### Professional Options
- **Vercel**: Professional hosting with GitHub integration
- **Netlify**: Full-featured hosting platform
- **Firebase Hosting**: Google's hosting solution

## 🛠️ Troubleshooting

### "File Not Found" Error
**Problem**: Raw GitHub URLs show "file not found" or don't work
**Solution**: Use Method 1 (right-click download) instead of trying to open raw URLs directly

### Files Don't Work When Downloaded
**Problem**: HTML files open but show errors or no interactivity
**Solution**: Make sure you downloaded BOTH files:
- `correlation_visualizer.html` AND `correlation_visualizer.js`
- Save them in the SAME folder
- The JavaScript file is required for interactivity

### Browser Security Warnings
**Problem**: Browser blocks local file access
**Solution**: 
- Try a different browser (Chrome usually works best)
- Or run a simple local server: `python -m http.server 8000` then open `http://localhost:8000`

### Visualizations Don't Load
**Problem**: Page loads but charts/interactive elements don't appear
**Solution**:
- Check browser console for JavaScript errors (F12)
- Ensure both HTML and JS files are in same folder
- Try refreshing the page (Ctrl+F5 or Cmd+Shift+R)

## 📝 Quick Start Checklist

- [ ] Use Method 1: Navigate to GitHub graphics folder
- [ ] Download `correlation_visualizer.html` (right-click Raw → Save as)
- [ ] Download `correlation_visualizer.js` (right-click Raw → Save as)
- [ ] Download `generate_static_graphics.html` (right-click Raw → Save as)
- [ ] Save all files in the same folder on your computer
- [ ] Double-click HTML files to open in browser
- [ ] Verify interactive features work (drag points, sliders, buttons)
- [ ] Explore all 5 visualization sections
- [ ] Try exporting graphics as PNG files

---

**Result**: Fully functional interactive visualizations running locally on your computer! 🎉

The visualizations are designed to work offline and provide the same educational value as any web-hosted version.