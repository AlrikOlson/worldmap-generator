class ProgressCallback:
    @staticmethod
    def notify(progress_callback, percentage, message):
        if progress_callback:
            progress_callback(percentage, message)
