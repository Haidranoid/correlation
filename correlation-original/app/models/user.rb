class User < ApplicationRecord
  # Include default devise modules. Others available are:
  # :confirmable, :lockable, :timeoutable, :trackable and :omniauthable
  devise :database_authenticatable, :trackable,
         :recoverable, :rememberable, :validatable
         
  # Add admin role functionality
  def admin?
    # For now, the first user or users with admin@admin.com are admins
    # You could add an admin boolean column later if needed
    email == 'admin@admin.com' || id == 1
  end
end
