import os
import sys
import pandas as pd
import librosa
import librosa.display
if os.environ.get("MPLBACKEND") is None:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

def output_settings():
    # 実行時間を使って出力ディレクトリを作成
    current_time = datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
    output_directory = os.path.join('./outputs', current_time)
    os.makedirs(output_directory, exist_ok=True)

    # ログの設定
    log_file_path = os.path.join(output_directory, 'training.log')

    # ロガーの作成
    logger = logging.getLogger()

    # ハンドラがすでに追加されていればスキップ
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # ハンドラの作成
    file_handler = logging.FileHandler(log_file_path)
    console_handler = logging.StreamHandler(sys.stdout)

    # フォーマットの設定
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # ハンドラをロガーに追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # output_dirをカスタム属性として追加
    logger.output_dir = output_directory

    return logger

# スペクトログラムの作成
def create_spectrogram(file_path, n_fft, hop_length, sampling_rate):
    y, sr = librosa.load(file_path, sr=sampling_rate)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    S_db = librosa.amplitude_to_db(np.abs(S))
    return S_db

# メルスペクトログラムの作成
def create_mel_spectrogram(file_path, n_fft, hop_length, sampling_rate, n_mels=128):
    y, sr = librosa.load(file_path, sr=sampling_rate)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=True)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

def prepare_dataloader(
        logger, hop_length, batch_size, data_csv_path, main_data_dir,
        n_fft=1024, data_num='loc1-6', mel='OFF',
        sampling_rate=16000, test_split=0.2, seed=42,
        representation='spectrogram'):
    """Create DataLoader objects for a specified audio representation.

    Parameters
    ----------
    logger : logging.Logger
        Logger configured via output_settings().
    hop_length : int
        Hop length for STFT when representation is ``spectrogram``.
    batch_size : int
        Batch size for the returned loaders.
    data_csv_path : str
        Path to metadata CSV describing audio locations.
    main_data_dir : str
        Root directory where the audio files are stored.
    n_fft : int, optional
        FFT size for STFT when representation is ``spectrogram``.
    data_num : str, optional
        Target location identifier (e.g. ``loc1`` or ``loc1-6``).
    mel : str, optional
        ``'ON'`` to create mel spectrograms instead of linear ones.
    sampling_rate : int, optional
        Sampling rate for librosa.load.
    test_split : float, optional
        Fraction of data used for the validation set.
    seed : int, optional
        Random seed used in train/validation split.
    representation : str, optional
        ``'spectrogram'`` (default) or ``'waveform'``.

    Returns
    -------
    Tuple[DataLoader, DataLoader, float, float]
        Train loader, validation loader, and min/max values prior to normalisation.
    """

    output_dir = logger.output_dir

    data = pd.read_csv(data_csv_path)

    if data_num == 'loc1-6':
        locations = [f'loc{i}' for i in range(1, 7)]
    else:
        locations = [f'{data_num}']

    features = []
    speeds = []
    vehicle_types_list = []
    directions_list = []
    locations_list = []

    sample_logged = False

    for loc in locations:
        loc_data = data[data['path'].str.contains(loc, regex=False)]

        for _, row in loc_data.iterrows():
            file_path = os.path.join(main_data_dir, row['path'])

            if representation == 'waveform':
                waveform, _ = librosa.load(file_path, sr=sampling_rate)
                features.append(waveform)

                if not sample_logged:
                    logger.info(
                        "sample_waveform: speed %s, %s, %s, %s",
                        row['speed'], loc, row['vehicle_type'], row['direction']
                    )
                    plt.figure(figsize=(10, 4))
                    librosa.display.waveshow(waveform, sr=sampling_rate)
                    plt.title('Waveform')
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'sample_waveform.png'))
                    plt.close()
                    sample_logged = True

            elif representation == 'spectrogram':
                if mel == 'ON':
                    feature = create_mel_spectrogram(
                        file_path,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        sampling_rate=sampling_rate
                    )
                else:
                    feature = create_spectrogram(
                        file_path,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        sampling_rate=sampling_rate
                    )

                features.append(feature)

                if not sample_logged:
                    logger.info(
                        "sample_spectrogram: speed %s, %s, %s, %s",
                        row['speed'], loc, row['vehicle_type'], row['direction']
                    )
                    plt.figure(figsize=(10, 4))
                    if mel == 'ON':
                        librosa.display.specshow(
                            feature,
                            sr=sampling_rate,
                            hop_length=hop_length,
                            x_axis='time',
                            y_axis='mel',
                            cmap='magma'
                        )
                    else:
                        librosa.display.specshow(
                            feature,
                            sr=sampling_rate,
                            hop_length=hop_length,
                            x_axis='time',
                            y_axis='hz',
                            cmap='magma'
                        )
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Spectrogram')
                    plt.xlabel("Time")
                    plt.ylabel("Frequency")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'sample_spectrogram.png'))
                    plt.close()

                    waveform, _ = librosa.load(file_path, sr=sampling_rate)
                    plt.figure(figsize=(10, 4))
                    librosa.display.waveshow(waveform, sr=sampling_rate)
                    plt.title('Waveform')
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'sample_waveform.png'))
                    plt.close()
                    sample_logged = True

            else:
                raise ValueError("representation must be either 'spectrogram' or 'waveform'.")

            speeds.append(row['speed'])
            vehicle_types_list.append(row['vehicle_type'])
            directions_list.append(row['direction'])
            locations_list.append(loc)

    if not features:
        raise ValueError("No audio data found. Please verify the CSV path and filters.")

    if representation == 'waveform':
        max_length = max(len(waveform) for waveform in features)
        padded = []
        for waveform in features:
            if len(waveform) < max_length:
                padded.append(np.pad(waveform, (0, max_length - len(waveform)), 'constant'))
            else:
                padded.append(waveform[:max_length])
        feature_array = np.array(padded, dtype=np.float32)
    else:
        feature_array = np.array(features, dtype=np.float32)

    speeds = np.array(speeds, dtype=np.float32)
    vehicle_types_list = np.array(vehicle_types_list)
    directions_list = np.array(directions_list)
    locations_list = np.array(locations_list)

    feature_min = float(feature_array.min())
    feature_max = float(feature_array.max())
    denom = feature_max - feature_min
    if denom <= 0:
        logger.warning("Feature max equals min; skipping normalisation scaling.")
        normalised_features = feature_array - feature_min
    else:
        normalised_features = (feature_array - feature_min) / denom

    split = train_test_split(
        normalised_features,
        speeds,
        vehicle_types_list,
        directions_list,
        locations_list,
        test_size=test_split,
        random_state=seed
    )

    (
        train_data,
        val_data,
        speeds_train,
        speeds_val,
        vehicle_train,
        vehicle_val,
        direction_train,
        direction_val,
        location_train,
        location_val
    ) = split

    vehicle_map = {'car': 0, 'cv': 1}
    direction_map = {'right': 0, 'left': 1}
    location_map = {
        'loc1': 0,
        'loc2': 1,
        'loc3': 2,
        'loc4': 3,
        'loc5': 4,
        'loc6': 5
    }

    vehicle_train_encoded = torch.tensor([vehicle_map.get(v, 0) for v in vehicle_train], dtype=torch.long)
    vehicle_val_encoded = torch.tensor([vehicle_map.get(v, 0) for v in vehicle_val], dtype=torch.long)
    direction_train_encoded = torch.tensor([direction_map.get(d, 0) for d in direction_train], dtype=torch.long)
    direction_val_encoded = torch.tensor([direction_map.get(d, 0) for d in direction_val], dtype=torch.long)
    location_train_encoded = torch.tensor([location_map.get(l, 0) for l in location_train], dtype=torch.long)
    location_val_encoded = torch.tensor([location_map.get(l, 0) for l in location_val], dtype=torch.long)

    train_tensors = [torch.tensor(sample, dtype=torch.float32).unsqueeze(0) for sample in train_data]
    val_tensors = [torch.tensor(sample, dtype=torch.float32).unsqueeze(0) for sample in val_data]
    train_speeds = torch.tensor(speeds_train, dtype=torch.float32)
    val_speeds = torch.tensor(speeds_val, dtype=torch.float32)

    train_dataset = TensorDataset(
        torch.stack(train_tensors),
        train_speeds,
        vehicle_train_encoded,
        direction_train_encoded,
        location_train_encoded
    )
    val_dataset = TensorDataset(
        torch.stack(val_tensors),
        val_speeds,
        vehicle_val_encoded,
        direction_val_encoded,
        location_val_encoded
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, feature_min, feature_max
