// Kinematic chains defining the connections between joints
const t2m_kinematic_chain = [
  [0, 2, 5, 8, 11], 
  [0, 1, 4, 7, 10], 
  [0, 3, 6, 9, 12, 15], 
  [9, 14, 17, 19, 21], 
  [9, 13, 16, 18, 20]
]

// Function to unpack the coordinates for Plotly
function unpackCoordinates(chain, coordinates) {
  return chain.map(index => coordinates[index]);
}

// Function to get te Kinematic Chain from a given coordinates in a given frame
function getKinematicChain(jointCoordinates, colores) {
  let traces = [];

  // Joints trace
  traces.push({
      x: jointCoordinates.x,
      z: jointCoordinates.y,
      y: jointCoordinates.z,
      mode: 'markers',
      marker: {
      size: 4,
      color: 'rgb(23, 190, 207)',
      line: {
        color: 'rgb(217, 217, 217)',
        width: 0.5
      },
      opacity: 0
      },
      type: 'scatter3d',
      name: 'Joints'
  });

  // Bones traces
  t2m_kinematic_chain.forEach((chain, index) => {
      traces.push({
        x: unpackCoordinates(chain, jointCoordinates.x),
        z: unpackCoordinates(chain, jointCoordinates.y),
        y: unpackCoordinates(chain, jointCoordinates.z),
        mode: 'lines',
        line: {
          width: 6,
          color: 'rgb' + colores
        },
        type: 'scatter3d',
        name: `Bone ${index + 1}`
      });
  });

  return traces;
}

// Function to get all the kinematic chains in all the frames of the motion
function getFrames(joints) {
  // Initialize an array to hold the frames
  let frames = [];

  // Loop through the frames of the motion data
  for (let i = 0; i < joints[0].Frames.length; i++) {
    // Define the joint coordinates for the first skeleton
    let traces1 = getKinematicChain(joints[0].Frames[i], '(255,0, 0)');
    
    // Define the joint coordinates for the second skeleton
    let traces2 = getKinematicChain(joints[1].Frames[i], '(0,0,255)');
    
    // Combine the traces for both skeletons
    let combinedTraces = traces1.concat(traces2);
    
    // Add the combined traces to the frames array
    frames.push({
      name: `frame${i}`,
      data: combinedTraces
    });
  }
  return frames
}

// Function to get the axis limits of the Layout
function getMinMaxLayout(joints) {
  // Initialize variables to store the min and max values for each axis
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  // Loop through each skeleton
  joints.forEach(skeleton => {
    // Loop through each frame of the skeleton
    skeleton.Frames.forEach(frame => {
      // Loop through each joint in the frame
      frame.x.forEach(joint => {
        // Update the min and max values for each axis
        minX = Math.min(minX, joint);
        maxX = Math.max(maxX, joint);
      });

      frame.y.forEach(joint => {
        minY = Math.min(minY, joint);
        maxY = Math.max(maxY, joint);
      });

      frame.z.forEach(joint => {
        minZ = Math.min(minZ, joint);
        maxZ = Math.max(maxZ, joint);
      });
    });
  });

  return [minX, maxX, minY, maxY, minZ, maxZ]
}

// Function to get the layout of the plot
function getLayout(joints) {
  // Initial layout configuration
  let layout = {
    showlegend: false,
    scene: {
      aspectmode: 'cube',
      aspectratio: {x: 1, y: 1, z: 1},
      xaxis: {      
        autorange: false,
        range: [-1, 1]
      },
      yaxis: {
        autorange: false,
        range: [-1, 1]
      }, 
      zaxis: {
        autorange: false,
        range: [0, 1.5]
      }, 
      camera: {
        eye: {x: 1.25, y: 1.25, z: 1.25}
      }
    },
    margin: {
      l: 0,
      r: 0,
      b: 0,
      t: 0
    },
  };

  // Now that we have the min and max values for each axis, we can set the range dynamically
  let [minX, maxX, minY, maxY, minZ, maxZ] = getMinMaxLayout(joints)  
  layout.scene.xaxis.range = [minX, maxX];
  layout.scene.yaxis.range = [minZ, maxZ];
  layout.scene.zaxis.range = [minY, maxY];

  return layout;
}

// Function to plot a whole motion
function plotMotion(HTMLid, HTMLbutton, joints) {
  var frames = getFrames(joints);
  var layout = getLayout(joints);

  // Initial data (first frame)
  let initialData = frames[0].data;

  let isPlaying = false; // Track the play/pause state
  let currentFrame = 0

  // Plot the initial frame
  Plotly.newPlot(HTMLid, initialData, layout).then(function() {
    // Add frames to the plot for animation
    Plotly.addFrames(HTMLid, frames);
  }).then(function() {
    document.getElementById(HTMLid).on('plotly_redraw', function(eventData) {
      currentFrame += 1;
    })
  });

  document.getElementById(HTMLbutton).addEventListener('click', function() {
      if (isPlaying) {
          // Pause the animation
          Plotly.animate(HTMLid, {
              data: [],
              layout: {},
              traces: [],
              config: {frameRedraw: false},
          }, {
              mode: 'next',
              transition: {duration: 0},
              frame: {duration: 0, redraw: true},
          });

          this.textContent = 'Play'; // Update button text
      } else {
          
          // Initialize frame counter when arriving to the begining
          if (currentFrame >= frames.length) {
            currentFrame = 0
          }

          if (currentFrame > 0) {
            frames_names = []
            for (let i = currentFrame; i < frames.length; i++) {
              frames_names.push(frames[i].name)
            }
            // Play the animation
            Plotly.animate(HTMLid, frames_names, {
                frame: {duration: 20, redraw: true},
                mode: 'immediate',
                transition: {duration: 0},
            });
          } else { 
            // Play the animation
            Plotly.animate(HTMLid, null, {
                frame: {duration: 20, redraw: true},
                mode: 'immediate',
                transition: {duration: 0},
            });
          }


          this.textContent = 'Pause'; // Update button text
      }

      isPlaying = !isPlaying; // Toggle state
  });




}