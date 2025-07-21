# yellow_blink_threshold_second: 10.0

# tracking.py

from collections import deque
from typing import Dict, List, Tuple

# ─── Configuración global ──────────────────────────────────────────────────────
# Estos valores vienen de `semantic.pb.txt`
# ventana de tiempo (segundos) para considerar la historia al decidir el color.
REVISE_TIME_S: float = 1.5
# si un amarillo dura menos que esto, se considera “blink” y no cambia de estado.
BLINK_THRESHOLD_S: float = 0.55
# cuántas veces consecutivas debe verse un nuevo color antes de aceptarlo.
HYSTERETIC_THRESHOLD_COUNT: int = 1
# ────────────────────────────────────────────────────────────────────────────────

class SemanticTable:
    """
    Estructura auxiliar que agrupa las detecciones
    de un mismo semáforo en un frame.
    """
    def __init__(self,
                 semantic_id: int,
                 time_stamp: float,
                 color: str):
        self.semantic_id = semantic_id
        self.time_stamp = time_stamp
        self.color = color
        self.last_bright_time = time_stamp
        self.last_dark_time = time_stamp
        self.blink = False
        self.hysteretic_color = color
        self.hysteretic_count = 0

class SemanticDecision:
    """
    Replica la lógica de `SemanticReviser` de Apollo:
      - Ventana temporal de revisión
      - Umbral de parpadeo
      - Histéresis de cambios
    """
    def __init__(self,
                 revise_time_s: float = REVISE_TIME_S,
                 blink_threshold_s: float = BLINK_THRESHOLD_S,
                 hysteretic_threshold: int = HYSTERETIC_THRESHOLD_COUNT):
        self.revise_time_s = revise_time_s
        self.blink_threshold_s = blink_threshold_s
        self.hysteretic_threshold = hysteretic_threshold
        # historial: semantic_id -> list de SemanticTable
        self.history: Dict[int, SemanticTable] = {}

    def update(self,
               frame_ts: float,
               assignments: List[Tuple[int,int]],
               recognitions: List[List[float]]
               ) -> Dict[int, Tuple[str,bool]]:
        """
        :param frame_ts: timestamp del frame actual en segundos
        :param assignments: lista de tuplas (proj_id, det_idx)
        :param recognitions: lista de scores [black, red, yellow, green] por det_idx
        :returns: dict {proj_id: (revised_color, blink_flag)}
        """
        # 1) Construir tablas semánticas por proj_id
        results: Dict[int, Tuple[str,bool]] = {}
        for proj_id, det_idx in assignments:
            # decidir color actual
            cls = int(max(range(len(recognitions[det_idx])),
                          key=lambda i: recognitions[det_idx][i]))
            color = ["black","red","yellow","green"][cls]

            # obtener o crear estado histórico
            if proj_id not in self.history:
                self.history[proj_id] = SemanticTable(proj_id, frame_ts, color)
            st = self.history[proj_id]

            # APOLLO'S HYSTERESIS LOGIC: Only when changing FROM black
            dt = frame_ts - st.time_stamp
            if color == "yellow" and dt < self.blink_threshold_s:
                # muñeco intermitente → no cambio de estado
                st.blink = True
            else:
                st.blink = False
                
                # Apply hysteresis ONLY when changing FROM black (unknown state)
                if st.color == "black":
                    # Conservative: need evidence to leave unknown state
                    if st.hysteretic_color == color:
                        st.hysteretic_count += 1
                    else:
                        st.hysteretic_color = color
                        st.hysteretic_count = 1
                    
                    # Only change FROM black with sufficient evidence
                    if st.hysteretic_count > self.hysteretic_threshold:
                        st.color = color
                        st.hysteretic_count = 0
                else:
                    # Between known states (red/green/yellow), update immediately
                    st.color = color
                    st.hysteretic_count = 0

            # actualizar timestamps
            st.time_stamp = frame_ts
            if color in ("red","green"):
                st.last_bright_time = frame_ts
            else:
                st.last_dark_time = frame_ts

            # después de revisar por tiempo, si pasa la ventana,
            # reiniciamos histéresis:
            if frame_ts - st.time_stamp > self.revise_time_s:
                st.hysteretic_count = 0

            results[proj_id] = (st.color, st.blink)

        return results

class TrafficLightTracker:
    """
    Envoltorio general que usa SemanticDecision.
    Puedes añadir aquí lógica adicional
    (por ejemplo, resetear estados tras N frames).
    """
    def __init__(self,
                 **semantic_kwargs):
        self.semantic = SemanticDecision(**semantic_kwargs)
        self.frame_counter = 0

    def track(self,
              frame_ts: float,
              assignments: List[Tuple[int,int]],
              recognitions: List[List[float]]
              ) -> Dict[int, Tuple[str,bool]]:
        """
        :param frame_ts: timestamp del frame
        :param assignments: misma interfaz que SemanticDecision
        :param recognitions: idem
        :returns: dict {proj_id: (revised_color, blink_flag)}
        """
        revised = self.semantic.update(frame_ts,
                                       assignments,
                                       recognitions)
        self.frame_counter += 1
        return revised
