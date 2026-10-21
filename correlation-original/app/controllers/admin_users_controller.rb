class AdminUsersController < ApplicationController
  before_action :ensure_admin!
  before_action :set_user, only: [:edit, :update, :destroy]

  def index
    @users = User.all.order(:email)
  end

  def new
    @user = User.new
  end

  def create
    @user = User.new(user_params)
    
    if @user.save
      redirect_to admin_users_path, notice: "User #{@user.email} was successfully created."
    else
      render :new, status: :unprocessable_entity
    end
  end

  def edit
  end

  def update
    # Handle password updates differently
    if params[:user][:password].present?
      if @user.update(user_params)
        redirect_to admin_users_path, notice: "User #{@user.email} was successfully updated."
      else
        render :edit, status: :unprocessable_entity
      end
    else
      # Update without password
      if @user.update(user_params.except(:password, :password_confirmation))
        redirect_to admin_users_path, notice: "User #{@user.email} was successfully updated."
      else
        render :edit, status: :unprocessable_entity
      end
    end
  end

  def destroy
    if @user == current_user
      redirect_to admin_users_path, alert: "You cannot delete your own account."
      return
    end
    
    email = @user.email
    @user.destroy
    redirect_to admin_users_path, notice: "User #{email} was successfully deleted."
  end

  private

  def ensure_admin!
    unless current_user&.admin?
      redirect_to root_path, alert: "Access denied. Admin privileges required."
    end
  end

  def set_user
    @user = User.find(params[:id])
  end

  def user_params
    params.require(:user).permit(:email, :password, :password_confirmation)
  end
end