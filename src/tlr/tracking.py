# yellow_blink_threshold_second: 10.0

# tracking.py

from collections import deque
from typing import Dict, List, Tuple

# ─── Configuración global ──────────────────────────────────────────────────────
# Estos valores vienen de `semantic.pb.txt` y semantic_decision.cc
# ventana de tiempo (segundos) para considerar la historia al decidir el color.
# Apollo semantic.pb.txt: revise_time_second: 1.5
REVISE_TIME_S: float = 1.5

# Umbral de blink (tiempo mínimo "dark" para detectar parpadeo)
# Apollo semantic.pb.txt: blink_threshold_second: 0.55
BLINK_THRESHOLD_S: float = 0.55

# cuántas veces consecutivas debe verse un nuevo color antes de aceptarlo.
# Apollo semantic.pb.txt: hysteretic_threshold_count: 1
# Condición Apollo: hysteretic_count > threshold (count > 1 → requiere 2 frames)
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
               recognitions: List[List[float]],
               projections: List = None
               ) -> Dict[str, Tuple[str,bool]]:
        """
        :param frame_ts: timestamp del frame actual en segundos
        :param assignments: lista de tuplas (proj_idx, det_idx)
        :param recognitions: lista de scores [black, red, yellow, green] por det_idx
        :param projections: lista de ProjectionROI objects con signal_id
        :returns: dict {signal_id: (revised_color, blink_flag)}
        """
        # 1) Construir tablas semánticas por signal_id
        results: Dict[str, Tuple[str,bool]] = {}
        for proj_idx, det_idx in assignments:
            # Obtener signal_id de la projection
            if projections and proj_idx < len(projections):
                signal_id = projections[proj_idx].signal_id
                if signal_id is None:
                    signal_id = f"unknown_{proj_idx}"
            else:
                # Fallback si no se pasan projections (retrocompatibilidad)
                signal_id = f"proj_{proj_idx}"
            # decidir color actual (como Apollo en ReviseBySemantic)
            cls = int(max(range(len(recognitions[det_idx])),
                          key=lambda i: recognitions[det_idx][i]))
            cur_color = ["black","red","yellow","green"][cls]

            # obtener o crear estado histórico POR SIGNAL_ID
            # Tracking sigue al semáforo físico, no a la projection box temporal
            if signal_id not in self.history:
                self.history[signal_id] = SemanticTable(signal_id, frame_ts, cur_color)
                st = self.history[signal_id]
                # Nuevo semáforo → guardar y continuar
                results[signal_id] = (st.color, st.blink)
                continue

            st = self.history[signal_id]

            # Calcular tiempo transcurrido
            dt = frame_ts - st.time_stamp

            # APOLLO TEMPORAL WINDOW CHECK (semantic_decision.cc:171-213)
            if dt <= self.revise_time_s:
                # DENTRO DE VENTANA TEMPORAL → Aplicar reglas de Apollo

                # APOLLO SWITCH STATEMENT por cur_color (semantic_decision.cc:174-208)
                if cur_color == "yellow":
                    # REGLA DE SECUENCIA TEMPORAL (Apollo semantic_decision.cc:176-182)
                    # "Because of the time sequence, yellow only exists after green and before red.
                    #  Any yellow after red is reset to red for the sake of safety until green displays."
                    if st.color == "red":
                        # YELLOW después de RED → INVÁLIDO, mantener RED
                        # ReviseLights mantiene iter->color (RED)
                        # NO cambia st.color
                        st.time_stamp = frame_ts
                        st.hysteretic_count = 0
                        st.blink = False
                    else:
                        # YELLOW después de GREEN/BLACK/UNKNOWN → VÁLIDO, aceptar
                        # UpdateHistoryAndLights actualiza iter->color a YELLOW
                        st.color = cur_color
                        st.time_stamp = frame_ts
                        st.last_dark_time = frame_ts  # Yellow es "dark"
                        st.hysteretic_count = 0
                        st.blink = False

                elif cur_color in ("red", "green"):
                    # CASE RED/GREEN (Apollo semantic_decision.cc:193-200)
                    # Alta confianza → aceptar cambio
                    st.color = cur_color
                    st.time_stamp = frame_ts
                    st.hysteretic_count = 0

                    # BLINK DETECTION (Apollo semantic_decision.cc:195-198)
                    # Detectar alternancia BRIGHT→DARK→BRIGHT
                    if (frame_ts - st.last_bright_time > self.blink_threshold_s and
                        st.last_dark_time > st.last_bright_time):
                        st.blink = True
                    else:
                        st.blink = False

                    # Actualizar timestamp de bright
                    st.last_bright_time = frame_ts

                elif cur_color == "black":
                    # CASE BLACK (Apollo semantic_decision.cc:202-208)
                    # Semáforo "apagado"
                    st.last_dark_time = frame_ts
                    st.hysteretic_count = 0  # BLACK resetea histéresis

                    if st.color in ("unknown", "black"):
                        # Ya estaba apagado/desconocido → aceptar BLACK
                        st.time_stamp = frame_ts
                        st.color = cur_color
                    else:
                        # Estaba encendido (RED/GREEN/YELLOW) → mantener color anterior
                        # NO actualizar st.color, NO actualizar st.time_stamp
                        pass
                    st.blink = False

                else:  # "unknown" o cualquier otro
                    # CASE UNKNOWN (Apollo semantic_decision.cc default)
                    # Baja confianza → mantener color anterior
                    # NO actualizar st.color, NO actualizar st.time_stamp
                    st.blink = False

            else:
                # VENTANA TEMPORAL EXPIRADA (>1.5s)
                # Apollo semantic_decision.cc:210-213
                # Resetear historial y aceptar color actual SIN validación
                st.time_stamp = frame_ts
                st.color = cur_color
                st.hysteretic_count = 0
                st.blink = False

                # Actualizar timestamps según el color
                if cur_color in ("red", "green"):
                    st.last_bright_time = frame_ts
                elif cur_color in ("yellow", "black"):
                    st.last_dark_time = frame_ts

            results[signal_id] = (st.color, st.blink)

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
              recognitions: List[List[float]],
              projections: List = None
              ) -> Dict[str, Tuple[str,bool]]:
        """
        :param frame_ts: timestamp del frame
        :param assignments: lista de (proj_idx, det_idx)
        :param recognitions: lista de scores por detección
        :param projections: lista de ProjectionROI con signal_id
        :returns: dict {signal_id: (revised_color, blink_flag)}
        """
        revised = self.semantic.update(frame_ts,
                                       assignments,
                                       recognitions,
                                       projections)
        self.frame_counter += 1
        return revised
