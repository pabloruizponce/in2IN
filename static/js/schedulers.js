// Define the exponential decay function and others in JavaScript
function weight(x, lambdaValue = 0.0075) {
    return x.map(x => Math.exp(-lambdaValue * (1000 - x)));
}

function weight3(x) {
    return x.map(x => 1 - ((1000 - x) / 1000));
}

function weight4(x, v) {
    return x.map(() => v);
}

function weight5(x, lambdaValue = 0.0075) {
    return x.map(x => 1 - Math.exp(-lambdaValue * (1000 - x)));
}

// Generate a range of counter values from 1000 to 0
let xValues = Array.from({length: 1000}, (_, i) => 1000 - i);

// Prepare data for plotting
let lambdaValues = [0.01, 0.00875, 0.0075, 0.00625, 0.005];
let constValues = [0, 0.25, 0.5, 0.75, 1];
let data = [];

// Add exponential decay plots
lambdaValues.forEach((lambdaValue, i) => {
    data.push({
        x: xValues,
        y: weight(xValues, lambdaValue),
        name: `exp:${lambdaValue}`,
        type: 'scatter',
        mode: 'lines',
        line: {color: `rgba(255,165,0,${0.5 + i * 0.1})`} // Example to mimic the colormap adjustment
    });
});

// Add inverse exponential plots
[0.01, 0.0075, 0.005].forEach((lambdaValue, i) => {
    data.push({
        x: xValues,
        y: weight5(xValues, lambdaValue),
        name: `inv-exp:${lambdaValue}`,
        type: 'scatter',
        mode: 'lines',
        line: {color: `rgba(0, 125, 209,${0.5 + i * 0.1})`} // Example to mimic the colormap adjustment
    });
});

// Add constant plots
constValues.forEach((v, i) => {
    data.push({
        x: xValues,
        y: weight4(xValues, v),
        name: `const:${v}`,
        type: 'scatter',
        mode: 'lines',
        line: {color: `rgba(0,128,0,${0.5 + i * 0.1})`} // Example to mimic the colormap adjustment
    });
});

// Add linear plot
data.push({
    x: xValues,
    y: weight3(xValues),
    name: 'lin',
    type: 'scatter',
    mode: 'lines',
    line: {color: 'rgba(252, 68, 240,1)'} // Example color
});



function plotScheduler(idName) {
    Plotly.newPlot(idName, data, {
        paper_bgcolor: '#F5F5F5', // Makes the outer background transparent
        plot_bgcolor: '#F5F5F5', // Makes the plot area background transparent
        height: 600, // Adjust this value as needed to fit the legend
        legend: {
          x: 1, // Adjust based on your layout preference
          y: 0.5, // Adjust based on your layout preference
          orientation: "v", // or "h" for horizontal
          itemsizing: "constant"
        },
        title: 'Weight Schedulers',
        xaxis: {
            title: 'Denoising Step',
            autorange: 'reversed',
            tickvals: [0, 250, 500, 750, 1000],
            ticktext: ['1000', '750', '500', '250', '0']
        },
        yaxis: {
            title: 'Weight'
        }
    });
}