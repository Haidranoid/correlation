require 'redcarpet'

module ApplicationHelper
  def render_markdown(content)
    return "" if content.blank?
    
    # Configure Redcarpet with options for proper markdown rendering
    renderer = Redcarpet::Render::HTML.new(
      filter_html: false,
      no_intra_emphasis: true,
      tables: true,
      fenced_code_blocks: true,
      autolink: true,
      disable_indented_code_blocks: false,
      strikethrough: true,
      lax_spacing: true,
      space_after_headers: true,
      superscript: true,
      underline: true,
      highlight: true,
      quote: true,
      footnotes: true
    )
    
    markdown = Redcarpet::Markdown.new(
      renderer,
      autolink: true,
      tables: true,
      fenced_code_blocks: true,
      strikethrough: true,
      lax_spacing: true,
      space_after_headers: true,
      superscript: true,
      underline: true,
      highlight: true,
      quote: true,
      footnotes: true,
      no_intra_emphasis: true,
      disable_indented_code_blocks: false
    )
    
    markdown.render(content).html_safe
  end
end
