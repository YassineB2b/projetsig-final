"""
View Routes

Renders HTML pages for the web dashboard.
"""

from flask import Blueprint, render_template

views_bp = Blueprint('views', __name__)


@views_bp.route('/')
def index():
    """Dashboard home page."""
    return render_template('index.html')


@views_bp.route('/upload')
def upload():
    """Image upload page."""
    return render_template('upload.html')


@views_bp.route('/live')
def live():
    """Live camera feed page."""
    return render_template('live.html')


@views_bp.route('/history')
def history():
    """Detection history page."""
    return render_template('history.html')


@views_bp.route('/settings')
def settings():
    """Settings page."""
    return render_template('settings.html')
