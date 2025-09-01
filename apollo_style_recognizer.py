import torch
import torch.nn.functional as F

class ApolloStyleRecognizer:
    """
    Implementación EXACTA del reconocimiento de Apollo
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # Apollo's exact color mapping
        self.status_map = ['black', 'red', 'yellow', 'green']
        
    def prob_to_color(self, output_probs):
        """
        Implementación exacta de Apollo's Prob2Color function
        """
        # output_probs es tensor [batch_size, 4] con probabilidades
        batch_size = output_probs.shape[0]
        results = []
        
        for i in range(batch_size):
            probs = output_probs[i]  # [4] vector
            
            # Find max probability and its index
            max_prob, max_idx = torch.max(probs, dim=0)
            
            # Apollo's logic: if max_prob > threshold, use max_idx, else use 0 (BLACK)
            if max_prob > self.threshold:
                color_id = max_idx.item()
            else:
                color_id = 0  # Force to BLACK
            
            # Create one-hot encoded result (like Apollo does)
            result = torch.zeros(4, device=output_probs.device)
            result[color_id] = 1.0
            results.append(result)
            
            # Debug info (like Apollo logs)
            print(f"Light status recognized as {self.status_map[color_id]}")
            print(f"Color Prob: {probs.tolist()}")
            print(f"Max prob: {max_prob:.4f}, threshold: {self.threshold}")
            
        return torch.stack(results)

    def recognize_batch(self, recognizer_model, input_batch):
        """
        Proceso completo de reconocimiento estilo Apollo
        """
        # 1. Get raw probabilities from model
        raw_output = recognizer_model(input_batch)
        
        # 2. Apply softmax (tu modelo ya lo hace)
        if not torch.allclose(torch.sum(raw_output, dim=1), torch.ones(raw_output.shape[0])):
            probs = F.softmax(raw_output, dim=1)
        else:
            probs = raw_output
            
        # 3. Apply Apollo's Prob2Color logic
        final_results = self.prob_to_color(probs)
        
        return final_results

# Como integrar esto en tu pipeline.py
def apollo_style_recognize(self, img, detections, tl_types):
    """
    Reemplazo para tu método recognize() actual
    """
    apollo_recognizer = ApolloStyleRecognizer(threshold=0.5)
    recognitions = []
    
    for detection, tl_type in zip(detections, tl_types):
        det_box = detection[1:5].type(torch.long)
        recognizer, shape = self.classifiers[tl_type-1]
        input = preprocess4rec(img, det_box, shape, self.means_rec)
        
        # Procesar con estilo Apollo
        input_batch = input.permute(2, 0, 1).unsqueeze(0)
        result = apollo_recognizer.recognize_batch(recognizer, input_batch)
        
        recognitions.append(result[0])  # Solo un item en el batch
        
    return torch.vstack(recognitions).reshape(-1, 4) if recognitions else torch.empty((0, 4), device=self.device)