"""Legacy RFMap detection methods; new code calls detection.detect_rf explicitly."""


class _RFDetectionCompatibility:
    __slots__ = ()

    @property
    def _rf_input_maps(self):
        return self._maps if self._rf_is_batch else (self,)

    def rf_2d(self, *args, **kwargs):
        """Compatibility array view; use detect_rf(self) for an RFResult."""
        from .detection import _legacy_rf_2d

        return _legacy_rf_2d(self, *args, **kwargs)

    def rf_1d(self, *args, **kwargs):
        """Compatibility detection/projection view; see detection.detect_rf."""
        from .detection import _legacy_rf_1d

        return _legacy_rf_1d(self, *args, **kwargs)

    def _detect_rf(self, trials, **options):
        from .detection import _detect_trials

        return _detect_trials(
            trials, maps=self._rf_input_maps, is_batch=self._rf_is_batch, **options,
        )
