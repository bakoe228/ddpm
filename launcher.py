import os
import sys
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext

class ScriptLauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Script Launcher")
        self.root.geometry("850x550")
        self.root.configure(bg="#1e1e1e")

        self.current_process = None
        self.log_queue = queue.Queue()

        self._setup_ui()
        self.refresh_scripts()
        self.root.after(100, self._process_log_queue)

    def _setup_ui(self):
        # Левая панель — список скриптов
        left_frame = tk.Frame(self.root, bg="#252526", width=220)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        tk.Label(
            left_frame, text="Скрипты (.py)", fg="#ffffff", bg="#252526", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self.script_listbox = tk.Listbox(
            left_frame, bg="#1e1e1e", fg="#d4d4d4", selectbackground="#007acc",
            selectforeground="#ffffff", bd=0, highlightthickness=1, font=("Consolas", 10)
        )
        self.script_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        btn_frame = tk.Frame(left_frame, bg="#252526")
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self.run_btn = tk.Button(
            btn_frame, text="▶ Запустить", bg="#0e639c", fg="#ffffff", activebackground="#1177bb",
            activeforeground="#ffffff", bd=0, font=("Segoe UI", 10, "bold"), pady=6, command=self.run_selected_script
        )
        self.run_btn.pack(fill=tk.X, pady=(0, 5))

        self.stop_btn = tk.Button(
            btn_frame, text="⏹ Остановить", bg="#a1260d", fg="#ffffff", activebackground="#be2d0f",
            activeforeground="#ffffff", bd=0, font=("Segoe UI", 10, "bold"), pady=6, command=self.stop_process, state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=(0, 5))

        tk.Button(
            btn_frame, text="🔄 Обновить список", bg="#3c3c3c", fg="#ffffff", bd=0,
            font=("Segoe UI", 9), pady=4, command=self.refresh_scripts
        ).pack(fill=tk.X)

        # Правая панель — Встроенная консоль
        right_frame = tk.Frame(self.root, bg="#1e1e1e")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(
            right_frame, text="Консоль вывода:", fg="#ffffff", bg="#1e1e1e", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(5, 5))

        self.console = scrolledtext.ScrolledText(
            right_frame, bg="#0c0c0c", fg="#cccccc", insertbackground="#ffffff",
            font=("Consolas", 10), bd=0, highlightthickness=1, highlightbackground="#333333"
        )
        self.console.pack(fill=tk.BOTH, expand=True)

        # Поле ввода команд в процесс (если скрипт просит input)
        input_frame = tk.Frame(right_frame, bg="#1e1e1e")
        input_frame.pack(fill=tk.X, pady=(5, 0))

        self.input_entry = tk.Entry(
            input_frame, bg="#252526", fg="#ffffff", insertbackground="#ffffff",
            bd=0, highlightthickness=1, font=("Consolas", 10)
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.input_entry.bind("<Return>", self.send_input)

        send_btn = tk.Button(
            input_frame, text="Отправить", bg="#3c3c3c", fg="#ffffff", bd=0,
            font=("Segoe UI", 9), padx=10, command=self.send_input
        )
        send_btn.pack(side=tk.RIGHT, padx=(5, 0))

    def refresh_scripts(self):
        self.script_listbox.delete(0, tk.END)
        folder = os.path.dirname(os.path.abspath(__file__))
        scripts = [f for f in os.listdir(folder) if f.endswith('.py') and f != os.path.basename(__file__)]
        for script in sorted(scripts):
            self.script_listbox.insert(tk.END, script)

    def log(self, text):
        self.log_queue.put(text)

    def _process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.console.insert(tk.END, msg)
            self.console.see(tk.END)
        self.root.after(100, self._process_log_queue)

    def run_selected_script(self):
        selection = self.script_listbox.curselection()
        if not selection:
            return

        script_name = self.script_listbox.get(selection[0])
        script_path = os.path.abspath(script_name)

        self.console.delete("1.0", tk.END)
        self.log(f"=== Запуск: {script_name} ===\n\n")

        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Запускаем скрипт в отдельном потоке с выключенным буфером вывода (-u)
        threading.Thread(target=self._execute_script, args=(script_path,), daemon=True).start()

    def _execute_script(self, script_path):
        try:
            self.current_process = subprocess.Popen(
                [sys.executable, "-u", script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            for line in iter(self.current_process.stdout.readline, ''):
                self.log(line)

            self.current_process.stdout.close()
            self.current_process.wait()
            self.log(f"\n=== Процесс завершен с кодом: {self.current_process.returncode} ===\n")

        except Exception as e:
            self.log(f"\n[Ошибка запуска]: {str(e)}\n")

        finally:
            self.current_process = None
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def send_input(self, event=None):
        if self.current_process and self.current_process.poll() is None:
            text = self.input_entry.get()
            self.input_entry.delete(0, tk.END)
            self.log(f"{text}\n")
            try:
                self.current_process.stdin.write(text + "\n")
                self.current_process.stdin.flush()
            except Exception as e:
                self.log(f"[Ошибка ввода]: {str(e)}\n")

    def stop_process(self):
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.log("\n=== Процесс принудительно остановлен ===\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ScriptLauncherApp(root)
    root.mainloop()