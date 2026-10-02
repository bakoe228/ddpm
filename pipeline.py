import sys
import subprocess
import time

def run_script(script_name):
    print(f"\n==================================================")
    print(f" [Pipeline] Запуск этапа: {script_name}")
    print(f" Время старта: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"==================================================\n")
    
    # Запускаем скрипт в том же окружении Python
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\n❌ [Ошибка] Скрипт {script_name} завершился с ошибкой (код {result.returncode}). Пайплайн остановлен!")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ [Успех] Этап {script_name} успешно завершен!")

def main():
    start_time = time.time()
    
    # 1. Запуск переразметки через Qwen2-VL
    run_script("generate_captions.py")
    
    # 2. Автоматический старт обучения сразу после разметки
    run_script("train.py")
    
    total_time = (time.time() - start_time) / 3600
    print(f"\n🎉 [Готово] Весь пайплайн (Разметка + Обучение) полностью выполнен за {total_time:.2f} часов!")

if __name__ == "__main__":
    main()