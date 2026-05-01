from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # 🔥 Dashboard / Index
    path('', views.index, name='index'),

    # 🍌 Banana
    path('banana/', views.Banana_Leaf, name='banana'),
    path('predict/', views.predict, name='predict'),

    # 🌿 Cinnamon
    path('cinnamon/', views.cinnamon_predict, name='cinnamon'),

    # 🔐 Auth
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('signup/', views.signup, name='signup'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/login/'), name='logout'),

    # (optional direct index URL)
    path('index/', views.index, name='index'),
]