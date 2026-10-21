class DocsController < ApplicationController
  def index
    @docs = [
      {
        title: "API Reference",
        description: "Complete guide to all API endpoints with examples and response formats",
        file: "API_REFERENCE.md",
        slug: "api-reference",
        icon: "📚"
      },
      {
        title: "Analysis Method",
        description: "Statistical methods and mathematical foundations with interactive visualizations",
        file: "ANALYSIS_METHOD.md",
        slug: "analysis-method",
        icon: "🔬"
      },
      {
        title: "Rake Task Reference",
        description: "All available rake tasks and commands for data processing and analysis",
        file: "RAKE_TASK_REFERENCE.md",
        slug: "rake-tasks",
        icon: "⚙️"
      },
      {
        title: "Known Shortcomings",
        description: "Current limitations and areas identified for future improvement",
        file: "KNOWN_SHORTCOMINGS.md",
        slug: "known-shortcomings",
        icon: "⚠️"
      }
    ]
  end

  def show
    @page = params[:page]
    @content = load_markdown_content(@page)
    
    if @content.nil?
      redirect_to documentation_path, alert: "Documentation page not found"
    end
  end

  private

  def load_markdown_content(page)
    file_map = {
      'api-reference' => 'docs/API_REFERENCE.md',
      'analysis-method' => 'docs/ANALYSIS_METHOD.md',
      'rake-tasks' => 'docs/RAKE_TASK_REFERENCE.md',
      'known-shortcomings' => 'docs/KNOWN_SHORTCOMINGS.md',
      'graphics-readme' => 'docs/graphics/README.md'
    }
    
    file_path = file_map[page]
    return nil unless file_path
    
    full_path = Rails.root.join(file_path)
    return nil unless File.exist?(full_path)
    
    File.read(full_path)
  end
end