class PrettyPrinter:
    # Extended ANSI escape codes
    STYLES = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'italic': '\033[3m',
        'underline': '\033[4m',
        'blink': '\033[5m',
        'inverse': '\033[7m',
        'hidden': '\033[8m',
        'strike': '\033[9m',
        
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        
        'bg_black': '\033[40m',
        'bg_red': '\033[41m',
        'bg_green': '\033[42m',
        'bg_yellow': '\033[43m',
        'bg_blue': '\033[44m',
        'bg_magenta': '\033[45m',
        'bg_cyan': '\033[46m',
        'bg_white': '\033[47m',
    }

    @classmethod
    def _style(cls, text, *styles):
        codes = ''.join([cls.STYLES[style] for style in styles])
        return f"{codes}{text}{cls.STYLES['reset']}"

    @classmethod
    def table(cls, headers, rows, title=None):
        # Create formatted table with borders
        col_width = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
        
        if title:
            total_width = sum(col_width) + 3*(len(headers)-1)
            print(cls._style(f"╒{'═'*(total_width)}╕", 'bold', 'blue'))
            print(cls._style(f"│ {title.center(total_width)} │", 'bold', 'blue'))
            print(cls._style(f"╞{'╪'.join('═'*w for w in col_width)}╡", 'bold', 'blue'))
        
        # Header
        header = cls._style("│ ", 'blue') + cls._style(" │ ", 'blue').join(
            cls._style(str(h).ljust(w), 'bold', 'white', 'bg_blue') 
            for h, w in zip(headers, col_width)
        ) + cls._style(" │", 'blue')
        print(header)
        
        # Separator
        print(cls._style(f"├{'┼'.join('─'*w for w in col_width)}┤", 'blue'))
        
        # Rows
        for row in rows:
            cells = []
            for item, w in zip(row, col_width):
                cell = cls._style(str(item).ljust(w), 'cyan')
                cells.append(cell)
            print(cls._style("│ ", 'blue') + cls._style(" │ ", 'blue').join(cells) + cls._style(" │", 'blue'))
        
        # Footer
        print(cls._style(f"╘{'╧'.join('═'*w for w in col_width)}╛", 'bold', 'blue'))

    @classmethod
    def _truncate_text(cls, text, max_length):
        """Truncate text with ellipsis if it exceeds max_length"""
        if len(text) <= max_length:
            return text
        # If we need to truncate, add an ellipsis
        if max_length > 3:
            return text[:max_length-3] + "..."
        return text[:max_length]

    @classmethod
    def section_header(cls, text):
        print("\n" + cls._style("╒═══════════════════════════════", 'bold', 'magenta'))
        print(cls._style(f"│ {text.upper()}", 'bold', 'magenta', 'italic'))
        print(cls._style("╘═══════════════════════════════", 'bold', 'magenta'))

    @classmethod
    def status(cls, label, message, status="info"):
        status_colors = {
            'info': ('blue', 'ℹ'),
            'success': ('green', '✔'),
            'warning': ('yellow', '⚠'),
            'error': ('red', '✖')
        }
        color, icon = status_colors.get(status, ('white', '○'))
        label_text = cls._style(f"[{label}]", 'bold', color)
        print(f"{cls._style(icon, color)} {label_text} {message}")

    @classmethod
    def code_block(cls, code, language="python"):
        print(cls._style(f"┏ {' ' + language + ' ':-^76} ┓", 'bold', 'white'))
        for line in code.split('\n'):
            print(cls._style("┃ ", 'white') + cls._style(f"{line:76}", 'cyan') + cls._style(" ┃", 'white'))
        print(cls._style(f"┗ {'':-^78} ┛", 'bold', 'white'))

    @classmethod
    def progress_bar(cls, current, total, label="Progress"):
        width = 50
        progress = current / total
        filled = int(width * progress)
        bar = cls._style("█" * filled, 'green') + cls._style("░" * (width - filled), 'dim')
        percent = cls._style(f"{progress:.0%}", 'bold', 'yellow')
        print(f"{label}: [{bar}] {percent} ({current}/{total})")
