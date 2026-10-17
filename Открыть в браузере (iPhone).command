#!/bin/bash
# Запускает Newton's Method WEB-приложение (для iPhone и браузера)

cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
    osascript -e 'display alert "Python 3 не найден" message "Установите Python 3 с python.org." as critical'
    exit 1
fi

# Устанавливаем зависимости
python3 -c "import numpy, matplotlib, flask" 2>/dev/null || {
    osascript -e 'display notification "Устанавливаются зависимости..." with title "Newton Web"'
    pip3 install numpy matplotlib flask --quiet
}

# Открываем браузер через секунду
(sleep 1.2 && open "http://localhost:5050") &

# Запускаем сервер
python3 newton_web.py
