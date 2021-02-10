// photos from flickr with creative commons license

var cy = cytoscape({
  container: document.getElementById('cy'),

  boxSelectionEnabled: false,
  autounselectify: true,

  style: cytoscape.stylesheet()
    .selector('node')
      .css({
        'height': 80,
        'width': 200,
        'background-fit': 'cover',
        'border-color': '#000',
        'border-width': 3,
        'border-opacity': 0.5
      })
    .selector('.eating')
      .css({
        'border-color': 'red'
      })
    .selector('.eater')
      .css({
        'border-width': 9
      })
    .selector('edge')
      .css({
        'curve-style': 'bezier',
        'width': 6,
        'target-arrow-shape': 'triangle',
        'line-color': '#ffaaaa',
        'target-arrow-color': '#ffaaaa'
      })
    .selector('#root_0')
      .css({
        'background-image': './root_0.png'
      })
    .selector('#root_0_0')
      .css({
        'background-image': './root_0_0.png'
      })
    .selector('#root_0_1')
      .css({
        'background-image': './root_0_1.png'
      })
    .selector('#root_1')
      .css({
        'background-image': './root_1.png'
      }),

  elements: {
    nodes: [
      { data: { id: 'root_0' } },
      { data: { id: 'root_0_0' } },
      { data: { id: 'root_0_1' } },
      { data: { id: 'root_1' } }
    ],
    edges: [
      { data: { source: 'root_0', target: 'root_0_0' } },
      { data: { source: 'root_0', target: 'root_0_1' } }
    ]
  },

  layout: {
    name: 'breadthfirst',
    directed: true,
    padding: 10
  }
}); // cy init

