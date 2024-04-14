import os
import subprocess
from flask import Flask, render_template, request, redirect, abort, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from data import db_session
from data.users import User
from forms.login import LoginForm
from forms.register import RegisterForm
from werkzeug.utils import secure_filename
import pandas as pd
import openpyxl


app = Flask(__name__)
app.config['SECRET_KEY'] = '48md27e492ry84580a2efd2fb7257aa8d'
UPLOAD_FOLDER = 'userdocs/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'xlsx'}
login_manager = LoginManager()
login_manager.init_app(app)
application = app


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def main():
    db_session.global_init('db/base.db')
    app.run()


def get_next_file_number(user_folder):
    if not os.path.exists(user_folder):
        return 1
    existing_files = os.listdir(user_folder)
    existing_numbers = [int(filename.split('.')[0]) for filename in existing_files if filename.split('.')[0].isdigit()]
    return max(existing_numbers) + 1 if existing_numbers else 1


@login_manager.user_loader
def load_user(user_id):
    db_sess = db_session.create_session()
    return db_sess.get(User, user_id)


@app.route('/', methods=['GET', 'POST'])
def visit():
    if current_user.is_authenticated:
        if request.method == 'POST':
            if 'file' not in request.files:
                print('Не могу прочитать файл')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                print('Нет выбранного файла')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
                if not os.path.exists(user_folder):
                    os.makedirs(user_folder)
                next_file_number = get_next_file_number(user_folder)
                new_filename = f"{next_file_number}.{filename.split('.')[-1]}"
                file_path = os.path.join(user_folder, new_filename)
                file.save(file_path)
                timestamp_column = request.form.get('timestamp_column')
                header_row = request.form.get('header_row')
                prediction_column = request.form.get('prediction_column')
                length = request.form.get('length')
                if None in [timestamp_column, header_row, prediction_column, length]:
                    print('Не все необходимые поля заполнены')
                    return redirect(request.url)
                lstm_script_path = 'lstm.py'
                subprocess.Popen(['python', lstm_script_path, file_path, user_folder, timestamp_column, header_row,
                                  prediction_column, length])

                return redirect(f'/results/{next_file_number}')
        return render_template('main.html')
    else:
        return redirect('/register')


@app.route("/register", methods=['GET', 'POST'])
def reqister():
    form = RegisterForm()
    if form.validate_on_submit():
        if form.password.data != form.password_again.data:
            return render_template('registration.html', title='Регистрация',
                                   form=form,
                                   message="Пароли не совпадают")
        db_sess = db_session.create_session()
        if db_sess.query(User).filter(User.email == form.email.data).first():
            return render_template('registration.html', title='Регистрация',
                                   form=form,
                                   message="Такой пользователь уже есть")
        user = User(
            name=form.name.data,
            email=form.email.data,
        )
        user.set_password(form.password.data)
        db_sess.add(user)
        db_sess.commit()
        return redirect('/login')
    return render_template('registration.html', title='Регистрация', form=form)


@app.route('/results/<int:id>')
@login_required
def results(id):
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    filename = os.path.join(user_folder, f"{id}.xlsx")
    if not os.path.exists(filename):
        abort(404)
    print(1)
    df = pd.read_excel(filename)
    print(2)
    cols = pd.read_excel(filename, nrows=0).columns.tolist()
    if cols != ['Timestamps', 'Predicted_Revenue']:
        return render_template('loading.html')
    timestamps = df['Timestamps'].tolist()
    predicted_revenue = df['Predicted_Revenue'].tolist()
    return render_template('results.html', timestamps=timestamps, predicted_revenue=predicted_revenue)


@app.route("/login")
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect('/')
    form = LoginForm()
    if form.validate_on_submit():
        db_sess = db_session.create_session()
        user = db_sess.query(User).filter(User.email == form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            return redirect("/")
        return render_template('login.html',
                               message="Неправильный логин или пароль",
                               form=form)
    return render_template('login.html', title='Авторизация', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect("/")


if __name__ == '__main__':
    main()
    app.run()