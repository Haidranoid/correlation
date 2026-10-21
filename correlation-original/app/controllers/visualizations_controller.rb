class VisualizationsController < ApplicationController
  def index
    @visualizations = [
      {
        title: "Correlation Visualizer",
        description: "Interactive demonstrations of correlation coefficient, time windows, and statistical significance",
        type: "interactive",
        icon: "📊",
        features: [
          "Drag data points to see real-time correlation changes",
          "Adjust time windows and offsets interactively", 
          "Explore Cohen's effect size guidelines",
          "Animated data processing workflow",
          "Parallel processing visualization"
        ]
      },
      {
        title: "Static Graphics Generator", 
        description: "Generate publication-ready diagrams and mathematical visualizations",
        type: "static",
        icon: "🖼️",
        features: [
          "Correlation formula with visual examples",
          "Statistical significance thresholds chart",
          "Time window analysis diagrams",
          "Data processing workflow diagrams", 
          "Export PNG images for presentations"
        ]
      }
    ]
  end

  def show
    @type = params[:type]
    
    case @type
    when 'interactive'
      @title = "Interactive Correlation Visualizer"
      @file_content = load_visualization_file('correlation_visualizer.html')
      @js_content = load_visualization_file('correlation_visualizer.js')
    when 'static'
      @title = "Static Graphics Generator"
      @file_content = load_visualization_file('generate_static_graphics.html')
    else
      redirect_to visualizations_path, alert: "Visualization not found"
      return
    end
    
    if @file_content.nil?
      redirect_to visualizations_path, alert: "Visualization files not found"
    end
  end

  private

  def load_visualization_file(filename)
    file_path = Rails.root.join('docs', 'graphics', filename)
    return nil unless File.exist?(file_path)
    
    content = File.read(file_path)
    
    # Extract only the body content if it's an HTML file
    if filename.end_with?('.html')
      # Extract content between <body> tags, removing the body tag itself
      if content =~ /<body[^>]*>(.*?)<\/body>/m
        content = $1
      end
    end
    
    content
  end
end