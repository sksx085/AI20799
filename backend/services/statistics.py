import threading

rumor_detection_lock = threading.Lock()
rumor_detection_count = 0
rumor_detection_history = []
MAX_HISTORY_DAYS = 30
