#!/bin/bash
# Запускает Newton's Method GUI
# Двойной клик в Finder — и программа откроется!

cd "$(dirname "$0")"

# Проверяем наличие Python 3
if ! command -v python3 &> /dev/null; then
    osascript -e 'display alert "Python 3 не найден" message "Установите Python 3 с python.org и попробуйте снова." as critical'
    exit 1
fi

# Устанавливаем зависимости если нужно
python3 -c "import numpy, matplotlib" 2>/dev/null || {
    osascript -e 'display notification "Устанавливаются зависимости (numpy, matplotlib)..." with title "Newton GUI"'
    pip3 install numpy matplotlib --quiet
}

# Запускаем GUI
python3 newton_gui.py
