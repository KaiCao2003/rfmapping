from spikeinterface.core import BaseRecording, BaseRecordingSegment


class DecimatedRecordingSegment(BaseRecordingSegment):
    
    def __init__(self, parent_segment, factor):
        self._parent = parent_segment
        self._factor = factor
        BaseRecordingSegment.__init__(self, **parent_segment.get_times_kwargs())
    
    def get_num_samples(self):
        return self._parent.get_num_samples() // self._factor
    
    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self._parent.get_traces(
            start_frame * self._factor,
            end_frame * self._factor,
            channel_indices
        )
        return traces[::self._factor, :]


class DecimatedRecording(BaseRecording):
    """
    Recording wrapper that decimates by selecting every k-th sample.
    No anti-aliasing filter. Preserves channel IDs, dtype, not metadata.
    """
    
    def __init__(self, parent_recording, decimation_factor):
        self._parent = parent_recording
        self._factor = int(decimation_factor)
        
        BaseRecording.__init__(
            self,
            sampling_frequency=parent_recording.get_sampling_frequency() / self._factor,
            channel_ids=parent_recording.get_channel_ids(),
            dtype=parent_recording.get_dtype()
        )
        
        for segment in parent_recording._recording_segments:
            self.add_recording_segment(DecimatedRecordingSegment(segment, self._factor))

        self._kwargs = {
            "parent_recording": parent_recording,
            "decimation_factor": decimation_factor,
        }
    
    def __repr__(self):
        return f"DecimatedRecording({self._parent}, factor={self._factor})"