from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('predict/', views.prediction_view, name='prediction'),
    path('historical/', views.historical_data, name='historical_data'),
    path('analysis/trend/', views.trend_analysis, name='trend_analysis'),
    path('data/update/', views.update_data, name='update_data'),

    path('password-reset/', 
         auth_views.PasswordResetView.as_view(
             template_name='stock_app/password_reset.html',
             email_template_name='stock_app/password_reset_email.html',
             subject_template_name='stock_app/password_reset_subject.txt'
         ),
         name='password_reset'),
    path('password-reset/done/', 
         auth_views.PasswordResetDoneView.as_view(
             template_name='stock_app/password_reset_done.html'
         ),
         name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/', 
         auth_views.PasswordResetConfirmView.as_view(
             template_name='stock_app/password_reset_confirm.html'
         ),
         name='password_reset_confirm'),
    path('password-reset-complete/', 
         auth_views.PasswordResetCompleteView.as_view(
             template_name='stock_app/password_reset_complete.html'
         ),
         name='password_reset_complete'),
    path('password-change/', 
         auth_views.PasswordChangeView.as_view(
             template_name='stock_app/password_change.html'
         ),
         name='password_change'),
    path('password-change/done/', 
         auth_views.PasswordChangeDoneView.as_view(
             template_name='stock_app/password_change_done.html'
         ),
         name='password_change_done'),
    
]