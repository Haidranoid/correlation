# 📊 Analysis Method Graphics and Visualizations

This directory contains interactive visualizations and static graphics generators to help understand the statistical methods and mathematical foundations of the Stock Correlation Analysis Platform.

> **📋 How to View These Files**: Since GitHub shows code instead of running HTML files, please see [HOW_TO_VIEW_VISUALIZATIONS.md](../../HOW_TO_VIEW_VISUALIZATIONS.md) for simple instructions to download and view these interactive demos locally.

## 🎯 Quick Start

### Interactive Visualizations
Open `correlation_visualizer.html` in your web browser to access:

1. **Correlation Coefficient Demo**
   - Drag data points to see correlation changes in real-time
   - Try preset patterns (perfect positive, negative, no correlation)
   - See significance thresholds update dynamically

2. **Time Window Analysis**
   - Adjust window sizes (1-30 days) for both symbols
   - Modify offset (-14 to +14 days) to explore lead-lag relationships
   - Visualize how parameters affect correlation analysis

3. **Statistical Significance Explorer**
   - Interactive Cohen's effect size guidelines
   - Real-time R² and variance explanation visualization
   - Sample size impact demonstration

4. **Data Processing Workflow**
   - Step-by-step animation of correlation calculation
   - Adjustable animation speed
   - Progress tracking through each processing stage

5. **Parallel Processing Architecture**
   - CPU core utilization visualization
   - Task distribution across cores
   - Performance metrics display

### Static Graphics Generator
Open `generate_static_graphics.html` to create publication-ready images:

- **Correlation Formula**: Mathematical formula with visual examples
- **Significance Chart**: Cohen's guidelines with effect size bands
- **Time Window Diagram**: Window and offset analysis illustration
- **Data Flow**: Complete processing workflow diagram
- **Parallel Architecture**: CPU core utilization and task distribution

## 📁 Files

### `correlation_visualizer.html`
Main interactive visualization dashboard with all demonstrations.

### `correlation_visualizer.js`
JavaScript implementation of all interactive visualizations including:
- Real-time correlation calculation
- Interactive data point manipulation
- Animation systems for workflow and parallel processing
- Export functionality for saving visualizations

### `generate_static_graphics.html`
Standalone graphics generator for creating static images suitable for:
- Documentation
- Presentations
- Academic papers
- Blog posts

### `README.md` (this file)
Documentation for using the visualization tools.

## 🚀 Usage Instructions

### For Learning and Understanding
1. **Start with Interactive Visualizer**: Open `correlation_visualizer.html`
2. **Follow the Analysis Method Document**: Use visualizations alongside [ANALYSIS_METHOD.md](../ANALYSIS_METHOD.md)
3. **Experiment with Parameters**: Try different correlation values, window sizes, and offsets
4. **Observe Real-time Changes**: See how modifications affect statistical significance

### For Documentation and Presentations
1. **Open Static Generator**: Launch `generate_static_graphics.html`
2. **Generate All Graphics**: Click "Generate All Graphics" button
3. **Export Individual Images**: Use "Export PNG" buttons for specific graphics
4. **Use in Documentation**: Include exported images in presentations or papers

## 🎨 Customization

### Modifying Visualizations
Edit `correlation_visualizer.js` to:
- Change color schemes
- Adjust animation speeds
- Add new visualization types
- Modify calculation parameters

### Adding New Graphics
Add new static graphics to `generate_static_graphics.html`:
- Create new canvas elements
- Implement drawing functions
- Add export functionality

## 📊 Educational Value

These visualizations help understand:

### Mathematical Concepts
- **Pearson Correlation**: How individual data points contribute to correlation coefficient
- **R-squared**: Visual representation of variance explanation
- **Effect Sizes**: Cohen's guidelines with practical interpretation

### Technical Implementation
- **Time Windows**: Impact of different analysis periods
- **Offset Analysis**: Lead-lag relationships in financial data
- **Parallel Processing**: How computational workload is distributed

### Statistical Significance
- **Thresholds**: Why certain values are considered significant
- **Sample Size**: Impact on statistical validity
- **Practical vs Statistical**: Difference between the two types of significance

## 🔧 Technical Requirements

### Minimum Requirements
- Modern web browser (Chrome, Firefox, Safari, Edge)
- JavaScript enabled
- HTML5 Canvas support

### Recommended Setup
- Large screen (1920x1080 or higher) for optimal visualization experience
- Chrome or Firefox for best performance
- Local web server for development (optional)

## 📖 Integration with Documentation

These visualizations are referenced throughout the main documentation:

- **[ANALYSIS_METHOD.md](../ANALYSIS_METHOD.md)**: Main technical documentation with embedded visualization links
- **[README.md](../../README.md)**: Platform overview with visual examples
- **[API_REFERENCE.md](../API_REFERENCE.md)**: API documentation with visual examples

## 🔄 Updates and Maintenance

### When to Update Graphics
- Mathematical formulas change
- New analysis methods are added
- UI/UX improvements are needed
- Bug fixes in calculations

### How to Update
1. Modify JavaScript in `correlation_visualizer.js`
2. Update HTML structure if needed
3. Test all interactive features
4. Regenerate static graphics if formulas change
5. Update documentation links if file names change

## 📈 Performance Notes

### Optimization Features
- **Canvas-based rendering** for smooth animations
- **Efficient calculation algorithms** for real-time updates
- **Memory management** for large datasets
- **Export functionality** without quality loss

### Browser Compatibility
- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+
- ❌ Internet Explorer (not supported)

---

**Built to enhance understanding of statistical correlation analysis through interactive learning and professional visualization tools.**