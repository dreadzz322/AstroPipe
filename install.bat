@echo off
chcp 65001 >nul <nul
setlocal
cd /d "%~dp0"

echo === AstroPipe: установка ===
where py >nul 2>nul
if errorlevel 1 (
    echo Не найден Python. Установите его с https://www.python.org/downloads/
    echo ^(при установке отметьте "Add python.exe to PATH"^) и запустите этот файл снова.
    pause
    exit /b 1
)

rem Берём обычную сборку Python проверенной версии; экспериментальные сборки не подходят.
set "PYV="
for %%V in (3.13 3.12 3.14 3.11 3.15) do if not defined PYV (
    py -%%V -c "import sys" >nul 2>nul && set "PYV=%%V"
)
if not defined PYV set "PYV=3"

if not exist ".venv\Scripts\python.exe" (
    echo Создаю окружение на Python %PYV%...
    py -%PYV% -m venv .venv || goto :fail
)

echo Ставлю библиотеки, это займёт несколько минут...
set "PIP=.venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check"
%PIP% --prefer-binary -r requirements.txt
if errorlevel 1 (
    echo Обычная установка не прошла, пробую обходной путь для очень новых версий Python...
    %PIP% --only-binary=:all: numpy scipy pillow pytest || goto :fail
    %PIP% pyyaml --config-settings="--global-option=--without-libyaml" || goto :fail
    %PIP% --only-binary=:all: astropy || goto :fail
    %PIP% astroquery || goto :fail
)

echo Проверяю, что всё работает...
".venv\Scripts\python.exe" -m pytest -q || goto :fail

echo.
echo Готово. Перетащите снимок или скан на файл run.bat.
pause
exit /b 0

:fail
echo.
echo Установка не удалась. Скопируйте текст выше и отправьте автору.
pause
exit /b 1
