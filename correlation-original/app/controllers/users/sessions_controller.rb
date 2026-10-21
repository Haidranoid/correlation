class Users::SessionsController < Devise::SessionsController
  def create
    super do |resource|
      if resource.persisted?
        # Force a full page reload after successful sign in
        response.headers['Turbo-Location'] = root_url
      end
    end
  end

  protected

  def after_sign_in_path_for(resource)
    root_path
  end

  def after_sign_out_path_for(resource_or_scope)
    new_user_session_path
  end
end
