def late_fusion_visual_audio(visual_probs, audio_probs, alpha=0.5):
    final_probs = {}
    for cls in visual_probs:
        v = visual_probs.get(cls, 0.0)
        a = audio_probs.get(cls, 0.0)
        final_probs[cls] = alpha * v + (1 - alpha) * a

    final_class = max(final_probs, key=final_probs.get)
    return final_class, final_probs
