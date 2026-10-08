from src.tracking.loader import load_player_tracks
from src.utils.config import load_config


config = load_config()

players = load_player_tracks(config.paths.tracks)

print(f"Tracked players: {len(players)}")

for player_id, player in list(players.items())[:10]:

    print(
        f"Player {player_id}: "
        f"{player.detection_count} detections | "
        f"frames {player.first_frame}-{player.last_frame}"
    )
