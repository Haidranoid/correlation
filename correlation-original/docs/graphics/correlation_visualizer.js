// Correlation Analysis Method Visualizations
// Interactive JavaScript for understanding correlation analysis concepts

// Global variables for managing state
let correlationData = [];
let isDragging = false;
let dragPointIndex = -1;
let animationFrameId = null;
let workflowStep = 0;
let parallelAnimationRunning = false;

// Initialize all visualizations when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeCorrelationVisualization();
    initializeTimeWindowVisualization();
    initializeStatisticalSignificance();
    initializeDataWorkflow();
    initializeParallelProcessing();
});

// 1. CORRELATION COEFFICIENT VISUALIZATION
function initializeCorrelationVisualization() {
    const canvas = document.getElementById('correlationCanvas');
    const ctx = canvas.getContext('2d');
    
    // Generate initial random data
    generateRandomData();
    
    // Add mouse event listeners for dragging
    canvas.addEventListener('mousedown', startDrag);
    canvas.addEventListener('mousemove', dragPoint);
    canvas.addEventListener('mouseup', stopDrag);
    canvas.addEventListener('mouseleave', stopDrag);
    
    function startDrag(e) {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Find closest point within 10 pixels
        for (let i = 0; i < correlationData.length; i++) {
            const px = correlationData[i].x * (canvas.width - 60) + 30;
            const py = (1 - correlationData[i].y) * (canvas.height - 60) + 30;
            
            if (Math.abs(x - px) < 10 && Math.abs(y - py) < 10) {
                isDragging = true;
                dragPointIndex = i;
                break;
            }
        }
    }
    
    function dragPoint(e) {
        if (!isDragging || dragPointIndex === -1) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Update point position (normalized to 0-1)
        correlationData[dragPointIndex].x = Math.max(0, Math.min(1, (x - 30) / (canvas.width - 60)));
        correlationData[dragPointIndex].y = Math.max(0, Math.min(1, 1 - (y - 30) / (canvas.height - 60)));
        
        drawCorrelationVisualization();
    }
    
    function stopDrag() {
        isDragging = false;
        dragPointIndex = -1;
    }
}

function generateRandomData() {
    correlationData = [];
    for (let i = 0; i < 20; i++) {
        correlationData.push({
            x: Math.random(),
            y: Math.random()
        });
    }
    drawCorrelationVisualization();
}

function generatePerfectPositive() {
    correlationData = [];
    for (let i = 0; i < 20; i++) {
        const x = i / 19;
        correlationData.push({
            x: x,
            y: x * 0.8 + 0.1 + (Math.random() - 0.5) * 0.02 // Small noise
        });
    }
    drawCorrelationVisualization();
}

function generatePerfectNegative() {
    correlationData = [];
    for (let i = 0; i < 20; i++) {
        const x = i / 19;
        correlationData.push({
            x: x,
            y: (1 - x) * 0.8 + 0.1 + (Math.random() - 0.5) * 0.02 // Small noise
        });
    }
    drawCorrelationVisualization();
}

function generateNoCorrelation() {
    correlationData = [];
    for (let i = 0; i < 20; i++) {
        correlationData.push({
            x: Math.random(),
            y: Math.random()
        });
    }
    drawCorrelationVisualization();
}

function drawCorrelationVisualization() {
    const canvas = document.getElementById('correlationCanvas');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw axes
    ctx.strokeStyle = '#2c3e50';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(30, 30);
    ctx.lineTo(30, canvas.height - 30);
    ctx.lineTo(canvas.width - 30, canvas.height - 30);
    ctx.stroke();
    
    // Draw axis labels
    ctx.fillStyle = '#2c3e50';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Variable X (Stock 1 % Change)', canvas.width / 2, canvas.height - 5);
    
    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Variable Y (Stock 2 % Change)', 0, 0);
    ctx.restore();
    
    // Calculate correlation
    const correlation = calculatePearsonCorrelation(correlationData);
    const rSquared = correlation * correlation;
    
    // Draw best fit line
    if (correlationData.length > 1) {
        const slope = calculateSlope(correlationData);
        const intercept = calculateIntercept(correlationData, slope);
        
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(30, (canvas.height - 60) * (1 - intercept) + 30);
        ctx.lineTo(canvas.width - 30, (canvas.height - 60) * (1 - (slope + intercept)) + 30);
        ctx.stroke();
    }
    
    // Draw data points
    ctx.fillStyle = '#e74c3c';
    correlationData.forEach(point => {
        const x = point.x * (canvas.width - 60) + 30;
        const y = (1 - point.y) * (canvas.height - 60) + 30;
        
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, 2 * Math.PI);
        ctx.fill();
    });
    
    // Update metrics
    document.getElementById('correlationValue').textContent = `r = ${correlation.toFixed(3)}`;
    document.getElementById('rSquaredValue').textContent = `R² = ${rSquared.toFixed(3)}`;
    document.getElementById('dataPoints').textContent = `n = ${correlationData.length}`;
    
    // Check significance
    const isSignificant = Math.abs(correlation) >= 0.3 && rSquared >= 0.09 && correlationData.length >= 10;
    const significanceEl = document.getElementById('significance');
    significanceEl.textContent = isSignificant ? 'Statistically Significant ⭐' : 'Not Significant';
    significanceEl.style.background = isSignificant ? '#27ae60' : '#e74c3c';
}

function calculatePearsonCorrelation(data) {
    if (data.length < 2) return 0;
    
    const n = data.length;
    const sumX = data.reduce((sum, point) => sum + point.x, 0);
    const sumY = data.reduce((sum, point) => sum + point.y, 0);
    const sumXY = data.reduce((sum, point) => sum + point.x * point.y, 0);
    const sumX2 = data.reduce((sum, point) => sum + point.x * point.x, 0);
    const sumY2 = data.reduce((sum, point) => sum + point.y * point.y, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
}

function calculateSlope(data) {
    const n = data.length;
    const sumX = data.reduce((sum, point) => sum + point.x, 0);
    const sumY = data.reduce((sum, point) => sum + point.y, 0);
    const sumXY = data.reduce((sum, point) => sum + point.x * point.y, 0);
    const sumX2 = data.reduce((sum, point) => sum + point.x * point.x, 0);
    
    return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
}

function calculateIntercept(data, slope) {
    const sumX = data.reduce((sum, point) => sum + point.x, 0);
    const sumY = data.reduce((sum, point) => sum + point.y, 0);
    const meanX = sumX / data.length;
    const meanY = sumY / data.length;
    
    return meanY - slope * meanX;
}

// 2. TIME WINDOW AND OFFSET ANALYSIS
function initializeTimeWindowVisualization() {
    updateTimeAnalysis();
}

function updateTimeAnalysis() {
    const window1 = parseInt(document.getElementById('window1').value);
    const window2 = parseInt(document.getElementById('window2').value);
    const offset = parseInt(document.getElementById('offset').value);
    
    // Update display values
    document.getElementById('window1Value').textContent = window1;
    document.getElementById('window2Value').textContent = window2;
    document.getElementById('offsetValue').textContent = offset;
    
    // Update analysis text
    document.getElementById('symbol1Analysis').textContent = `${window1}-day percentage changes`;
    document.getElementById('symbol2Analysis').textContent = `${window2}-day percentage changes`;
    
    let offsetText, interpretationText;
    if (offset > 0) {
        offsetText = `${offset} days forward (predictive)`;
        interpretationText = `Symbol 1 movements predict Symbol 2 movements ${offset} days later`;
    } else if (offset < 0) {
        offsetText = `${Math.abs(offset)} days backward (lagging)`;
        interpretationText = `Symbol 1 reacts to Symbol 2 movements from ${Math.abs(offset)} days ago`;
    } else {
        offsetText = 'Contemporaneous (same time)';
        interpretationText = 'Direct relationship between current movements';
    }
    
    document.getElementById('offsetAnalysis').textContent = offsetText;
    document.getElementById('interpretationText').textContent = interpretationText;
    
    drawTimeWindowVisualization(window1, window2, offset);
}

function drawTimeWindowVisualization(window1, window2, offset) {
    const canvas = document.getElementById('timeWindowCanvas');
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Generate sample price data
    const days = 60;
    const symbol1Data = generateSamplePriceData(days, 100, 0.02);
    const symbol2Data = generateSamplePriceData(days, 120, 0.025);
    
    // Draw time axis
    ctx.strokeStyle = '#2c3e50';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(50, canvas.height - 50);
    ctx.lineTo(canvas.width - 50, canvas.height - 50);
    ctx.stroke();
    
    // Draw price data
    drawPriceSeries(ctx, symbol1Data, '#e74c3c', 'Symbol 1', 100, 200);
    drawPriceSeries(ctx, symbol2Data, '#3498db', 'Symbol 2', 250, 350);
    
    // Draw analysis windows
    drawAnalysisWindows(ctx, window1, window2, offset, days);
    
    // Draw legend
    drawTimeWindowLegend(ctx);
}

function generateSamplePriceData(days, startPrice, volatility) {
    const data = [startPrice];
    for (let i = 1; i < days; i++) {
        const change = (Math.random() - 0.5) * volatility * startPrice;
        const newPrice = Math.max(startPrice * 0.5, data[i-1] + change);
        data.push(newPrice);
    }
    return data;
}

function drawPriceSeries(ctx, data, color, label, minY, maxY) {
    const canvas = ctx.canvas;
    const minPrice = Math.min(...data);
    const maxPrice = Math.max(...data);
    const priceRange = maxPrice - minPrice;
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    data.forEach((price, index) => {
        const x = 50 + (index / (data.length - 1)) * (canvas.width - 100);
        const y = maxY - ((price - minPrice) / priceRange) * (maxY - minY);
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
    
    // Add label
    ctx.fillStyle = color;
    ctx.font = '14px Arial';
    ctx.fillText(label, 60, minY - 10);
}

function drawAnalysisWindows(ctx, window1, window2, offset, totalDays) {
    const canvas = ctx.canvas;
    const windowStart = Math.floor(totalDays * 0.4);
    const windowWidth = Math.max(window1, window2);
    
    // Draw analysis window overlay
    ctx.fillStyle = 'rgba(243, 156, 18, 0.3)';
    const startX = 50 + (windowStart / totalDays) * (canvas.width - 100);
    const endX = 50 + ((windowStart + windowWidth) / totalDays) * (canvas.width - 100);
    
    ctx.fillRect(startX, 100, endX - startX, 300);
    
    // Draw offset indicator
    if (offset !== 0) {
        const offsetX = 50 + ((windowStart + offset) / totalDays) * (canvas.width - 100);
        ctx.strokeStyle = '#f39c12';
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(offsetX, 100);
        ctx.lineTo(offsetX, 400);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Add offset label
        ctx.fillStyle = '#f39c12';
        ctx.font = '12px Arial';
        ctx.fillText(`Offset: ${offset}d`, offsetX + 5, 120);
    }
}

function drawTimeWindowLegend(ctx) {
    // This is handled by the HTML legend below the canvas
}

// 3. STATISTICAL SIGNIFICANCE
function initializeStatisticalSignificance() {
    updateSignificance();
}

function updateSignificance() {
    const correlation = parseFloat(document.getElementById('corrSlider').value) / 100;
    const sampleSize = parseInt(document.getElementById('sampleSlider').value);
    
    document.getElementById('corrSliderValue').textContent = correlation.toFixed(2);
    document.getElementById('sampleSliderValue').textContent = sampleSize;
    
    drawSignificanceVisualization(correlation, sampleSize);
}

function drawSignificanceVisualization(correlation, sampleSize) {
    const canvas = document.getElementById('significanceCanvas');
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw Cohen's effect size ranges
    const ranges = [
        { min: 0, max: 0.1, color: '#bdc3c7', label: 'No Effect' },
        { min: 0.1, max: 0.3, color: '#f39c12', label: 'Small Effect' },
        { min: 0.3, max: 0.5, color: '#e67e22', label: 'Medium Effect ⭐' },
        { min: 0.5, max: 0.7, color: '#e74c3c', label: 'Large Effect ⭐' },
        { min: 0.7, max: 0.9, color: '#c0392b', label: 'Very Large Effect ⭐' },
        { min: 0.9, max: 1.0, color: '#8e44ad', label: 'Extremely Large ⭐' }
    ];
    
    // Draw effect size bands
    const bandHeight = 50;
    const startY = 50;
    
    ranges.forEach((range, index) => {
        const x = 50 + range.min * (canvas.width - 100);
        const width = (range.max - range.min) * (canvas.width - 100);
        
        ctx.fillStyle = range.color;
        ctx.fillRect(x, startY, width, bandHeight);
        
        // Add labels
        ctx.fillStyle = '#2c3e50';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(range.label, x + width/2, startY + bandHeight + 20);
        ctx.fillText(range.min.toFixed(1), x, startY + bandHeight + 35);
        if (index === ranges.length - 1) {
            ctx.fillText(range.max.toFixed(1), x + width, startY + bandHeight + 35);
        }
    });
    
    // Draw current correlation marker
    const currentX = 50 + Math.abs(correlation) * (canvas.width - 100);
    ctx.fillStyle = '#2c3e50';
    ctx.beginPath();
    ctx.moveTo(currentX, startY - 10);
    ctx.lineTo(currentX - 10, startY - 25);
    ctx.lineTo(currentX + 10, startY - 25);
    ctx.closePath();
    ctx.fill();
    
    // Draw R² visualization
    const rSquared = correlation * correlation;
    drawRSquaredVisualization(ctx, rSquared, 150);
    
    // Draw sample size requirements
    drawSampleSizeVisualization(ctx, sampleSize, 250);
    
    // Update metrics
    updateSignificanceMetrics(correlation, rSquared, sampleSize);
}

function drawRSquaredVisualization(ctx, rSquared, y) {
    const canvas = ctx.canvas;
    const boxSize = 100;
    const x = canvas.width / 2 - boxSize / 2;
    
    // Draw total variance box
    ctx.strokeStyle = '#2c3e50';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, boxSize, boxSize);
    
    // Draw explained variance
    ctx.fillStyle = '#3498db';
    ctx.fillRect(x, y, boxSize, boxSize * rSquared);
    
    // Draw unexplained variance
    ctx.fillStyle = '#ecf0f1';
    ctx.fillRect(x, y + boxSize * rSquared, boxSize, boxSize * (1 - rSquared));
    
    // Add labels
    ctx.fillStyle = '#2c3e50';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Variance Explained', x + boxSize/2, y - 10);
    ctx.fillText(`R² = ${rSquared.toFixed(3)}`, x + boxSize/2, y + boxSize + 20);
    ctx.fillText(`${(rSquared * 100).toFixed(1)}%`, x + boxSize/2, y + boxSize/2);
}

function drawSampleSizeVisualization(ctx, sampleSize, y) {
    const canvas = ctx.canvas;
    const maxSize = 100;
    const dotSize = 4;
    const cols = 10;
    const rows = Math.ceil(maxSize / cols);
    
    const startX = canvas.width / 2 - (cols * dotSize * 2) / 2;
    
    for (let i = 0; i < maxSize; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        const x = startX + col * dotSize * 2;
        const dotY = y + row * dotSize * 2;
        
        ctx.fillStyle = i < sampleSize ? '#27ae60' : '#ecf0f1';
        ctx.beginPath();
        ctx.arc(x, dotY, dotSize/2, 0, 2 * Math.PI);
        ctx.fill();
    }
    
    // Add label
    ctx.fillStyle = '#2c3e50';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`Sample Size: ${sampleSize}`, canvas.width / 2, y - 10);
    ctx.fillText(sampleSize >= 10 ? 'Adequate ✓' : 'Too Small ✗', canvas.width / 2, y + rows * dotSize * 2 + 20);
}

function updateSignificanceMetrics(correlation, rSquared, sampleSize) {
    document.getElementById('currentCorr').textContent = `r = ${correlation.toFixed(3)}`;
    document.getElementById('currentRSquared').textContent = `R² = ${rSquared.toFixed(3)}`;
    document.getElementById('currentSample').textContent = `n = ${sampleSize}`;
    
    // Determine effect size
    const absCorr = Math.abs(correlation);
    let effectSize;
    if (absCorr < 0.1) effectSize = 'No Effect';
    else if (absCorr < 0.3) effectSize = 'Small Effect';
    else if (absCorr < 0.5) effectSize = 'Medium Effect';
    else if (absCorr < 0.7) effectSize = 'Large Effect';
    else if (absCorr < 0.9) effectSize = 'Very Large Effect';
    else effectSize = 'Extremely Large';
    
    document.getElementById('effectSize').textContent = effectSize;
    
    // Determine if saved
    const isSignificant = absCorr >= 0.3 && rSquared >= 0.09 && sampleSize >= 10;
    const saveEl = document.getElementById('saveStatus');
    saveEl.textContent = isSignificant ? 'Saved to Database ⭐' : 'Not Saved';
    saveEl.style.background = isSignificant ? '#27ae60' : '#e74c3c';
}

// 4. DATA PROCESSING WORKFLOW
function initializeDataWorkflow() {
    resetWorkflow();
}

function startWorkflowAnimation() {
    resetWorkflow();
    workflowStep = 0;
    animateWorkflowStep();
}

function resetWorkflow() {
    workflowStep = 0;
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    drawWorkflowVisualization();
    document.getElementById('currentStep').textContent = 'Ready to start';
    document.getElementById('progressText').textContent = '0% complete';
}

function animateWorkflowStep() {
    const steps = [
        'Loading raw stock price data...',
        'Calculating percentage changes...',
        'Applying time windows...',
        'Aligning data points with offset...',
        'Computing Pearson correlation...',
        'Calculating R-squared...',
        'Checking statistical significance...',
        'Saving results to database...'
    ];
    
    if (workflowStep < steps.length) {
        document.getElementById('currentStep').textContent = steps[workflowStep];
        document.getElementById('progressText').textContent = `${Math.round((workflowStep + 1) / steps.length * 100)}% complete`;
        
        drawWorkflowVisualization();
        
        const speed = parseInt(document.getElementById('animationSpeed').value);
        const delay = 1000 / speed;
        
        setTimeout(() => {
            workflowStep++;
            animateWorkflowStep();
        }, delay);
    } else {
        document.getElementById('currentStep').textContent = 'Analysis complete!';
        document.getElementById('progressText').textContent = '100% complete';
    }
}

function drawWorkflowVisualization() {
    const canvas = document.getElementById('workflowCanvas');
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const steps = [
        { name: 'Raw Data', x: 100, y: 100, width: 120, height: 80 },
        { name: '% Changes', x: 300, y: 100, width: 120, height: 80 },
        { name: 'Time Windows', x: 500, y: 100, width: 120, height: 80 },
        { name: 'Alignment', x: 100, y: 250, width: 120, height: 80 },
        { name: 'Correlation', x: 300, y: 250, width: 120, height: 80 },
        { name: 'R-squared', x: 500, y: 250, width: 120, height: 80 },
        { name: 'Significance', x: 200, y: 400, width: 120, height: 80 },
        { name: 'Database', x: 400, y: 400, width: 120, height: 80 }
    ];
    
    // Draw workflow steps
    steps.forEach((step, index) => {
        const isActive = index <= workflowStep;
        const isCompleted = index < workflowStep;
        
        // Draw step box
        ctx.fillStyle = isCompleted ? '#27ae60' : isActive ? '#3498db' : '#ecf0f1';
        ctx.fillRect(step.x, step.y, step.width, step.height);
        
        // Draw step border
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 2;
        ctx.strokeRect(step.x, step.y, step.width, step.height);
        
        // Draw step label
        ctx.fillStyle = isActive || isCompleted ? '#ffffff' : '#2c3e50';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(step.name, step.x + step.width/2, step.y + step.height/2 + 5);
        
        // Draw checkmark for completed steps
        if (isCompleted) {
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(step.x + step.width - 25, step.y + 15);
            ctx.lineTo(step.x + step.width - 15, step.y + 25);
            ctx.lineTo(step.x + step.width - 5, step.y + 10);
            ctx.stroke();
        }
    });
    
    // Draw arrows
    drawWorkflowArrows(ctx, steps);
}

function drawWorkflowArrows(ctx, steps) {
    const arrows = [
        [0, 1], [1, 2], [2, 5], [5, 4], [4, 3], [3, 6], [6, 7]
    ];
    
    ctx.strokeStyle = workflowStep > 0 ? '#3498db' : '#bdc3c7';
    ctx.lineWidth = 2;
    
    arrows.forEach(([from, to], index) => {
        const isActive = index < workflowStep;
        ctx.strokeStyle = isActive ? '#27ae60' : '#bdc3c7';
        
        const fromStep = steps[from];
        const toStep = steps[to];
        
        const fromX = fromStep.x + fromStep.width/2;
        const fromY = fromStep.y + fromStep.height/2;
        const toX = toStep.x + toStep.width/2; 
        const toY = toStep.y + toStep.height/2;
        
        // Draw arrow line
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
        
        // Draw arrowhead
        const angle = Math.atan2(toY - fromY, toX - fromX);
        const headLength = 10;
        
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI/6), toY - headLength * Math.sin(angle - Math.PI/6));
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI/6), toY - headLength * Math.sin(angle + Math.PI/6));
        ctx.stroke();
    });
}

// 5. PARALLEL PROCESSING ARCHITECTURE
function initializeParallelProcessing() {
    updateParallelVisualization();
}

function updateParallelVisualization() {
    const coreCount = parseInt(document.getElementById('coreCount').value);
    const taskCount = parseInt(document.getElementById('taskCount').value);
    
    document.getElementById('taskCountValue').textContent = taskCount;
    
    drawParallelVisualization(coreCount, taskCount);
}

function startParallelAnimation() {
    if (parallelAnimationRunning) return;
    
    parallelAnimationRunning = true;
    animateParallelProcessing();
}

function animateParallelProcessing() {
    const coreCount = parseInt(document.getElementById('coreCount').value);
    const taskCount = parseInt(document.getElementById('taskCount').value);
    
    let completedTasks = 0;
    const startTime = Date.now();
    
    function updateProgress() {
        if (completedTasks < taskCount && parallelAnimationRunning) {
            completedTasks += Math.min(coreCount, taskCount - completedTasks);
            
            const elapsedTime = (Date.now() - startTime) / 1000;
            const throughput = completedTasks / elapsedTime;
            const utilization = Math.min(100, (completedTasks / taskCount) * 100);
            const speedup = Math.min(coreCount, throughput / (taskCount / (elapsedTime || 1)));
            const efficiency = (speedup / coreCount) * 100;
            
            document.getElementById('coreUtilization').textContent = `Core Utilization: ${utilization.toFixed(1)}%`;
            document.getElementById('throughput').textContent = `Throughput: ${throughput.toFixed(1)} tasks/sec`;
            document.getElementById('speedup').textContent = `Speedup: ${speedup.toFixed(1)}x`;
            document.getElementById('efficiency').textContent = `Efficiency: ${efficiency.toFixed(1)}%`;
            
            drawParallelVisualization(coreCount, taskCount, completedTasks);
            
            setTimeout(updateProgress, 100);
        } else {
            parallelAnimationRunning = false;
        }
    }
    
    updateProgress();
}

function drawParallelVisualization(coreCount, taskCount, completedTasks = 0) {
    const canvas = document.getElementById('parallelCanvas');
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw CPU cores
    const coreWidth = 80;
    const coreHeight = 60;
    const coreSpacing = 20;
    const coresPerRow = 4;
    const totalCoreWidth = coresPerRow * (coreWidth + coreSpacing) - coreSpacing;
    const coreStartX = (canvas.width - totalCoreWidth) / 2;
    
    for (let i = 0; i < coreCount; i++) {
        const row = Math.floor(i / coresPerRow);
        const col = i % coresPerRow;
        const x = coreStartX + col * (coreWidth + coreSpacing);
        const y = 50 + row * (coreHeight + coreSpacing);
        
        // Determine core activity
        const isActive = completedTasks > 0 && i < Math.min(coreCount, taskCount - completedTasks + coreCount);
        
        // Draw core
        ctx.fillStyle = isActive ? '#e74c3c' : '#3498db';
        ctx.fillRect(x, y, coreWidth, coreHeight);
        
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, coreWidth, coreHeight);
        
        // Draw core label
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Core ${i + 1}`, x + coreWidth/2, y + coreHeight/2);
        
        // Draw activity indicator
        if (isActive) {
            ctx.fillStyle = '#f39c12';
            ctx.beginPath();
            ctx.arc(x + coreWidth - 10, y + 10, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
    
    // Draw task queue
    const queueY = 250;
    const taskSize = 15;
    const tasksPerRow = Math.floor((canvas.width - 100) / (taskSize + 2));
    
    ctx.fillStyle = '#2c3e50';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Task Queue', canvas.width / 2, queueY - 20);
    
    for (let i = 0; i < taskCount; i++) {
        const row = Math.floor(i / tasksPerRow);
        const col = i % tasksPerRow;
        const x = 50 + col * (taskSize + 2);
        const y = queueY + row * (taskSize + 2);
        
        const isCompleted = i < completedTasks;
        const isProcessing = i >= completedTasks && i < completedTasks + coreCount;
        
        ctx.fillStyle = isCompleted ? '#27ae60' : isProcessing ? '#f39c12' : '#ecf0f1';
        ctx.fillRect(x, y, taskSize, taskSize);
        
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, taskSize, taskSize);
    }
    
    // Draw legend
    drawParallelLegend(ctx);
}

function drawParallelLegend(ctx) {
    const legendItems = [
        { color: '#3498db', label: 'Idle Core' },
        { color: '#e74c3c', label: 'Active Core' },
        { color: '#ecf0f1', label: 'Pending Task' },
        { color: '#f39c12', label: 'Processing Task' },
        { color: '#27ae60', label: 'Completed Task' }
    ];
    
    const legendY = 400;
    const itemWidth = 120;
    const startX = (ctx.canvas.width - legendItems.length * itemWidth) / 2;
    
    legendItems.forEach((item, index) => {
        const x = startX + index * itemWidth;
        
        // Draw color box
        ctx.fillStyle = item.color;
        ctx.fillRect(x, legendY, 15, 15);
        
        // Draw label
        ctx.fillStyle = '#2c3e50';
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(item.label, x + 20, legendY + 12);
    });
}

// EXPORT FUNCTIONALITY
function exportCanvas(canvasId, filename) {
    const canvas = document.getElementById(canvasId);
    const link = document.createElement('a');
    link.download = filename;
    link.href = canvas.toDataURL('image/png');
    link.click();
}

// Update speed display
document.getElementById('animationSpeed').addEventListener('input', function() {
    document.getElementById('speedValue').textContent = this.value;
});