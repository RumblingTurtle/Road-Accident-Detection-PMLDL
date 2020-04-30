# vim: expandtab:ts=4:sw=4
import math


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self.vel = []
        self.acc = []
        self._n_init = n_init
        self._max_age = max_age
        self.avg_pos = []
        self.pos_pool = []
        self.vel_q = []
        self.acc_q = []
        self.center = None
        self.collision_status = 0

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, min y, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        coordinates = detection.to_xyah()
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, coordinates)
        self.features.append(detection.feature)
        self.center = coordinates[:2]
        if self.hits > 5:
            avg = (sum([x for x, _ in self.pos_pool]) / 5, sum([y for _, y in self.pos_pool]) / 5)
            self.avg_pos.insert(0, avg)
            self.pos_pool.insert(0, coordinates[:2])
        else:
            self.pos_pool.insert(0, coordinates[:2])
        if len(self.avg_pos) > 2:
            self._calc_vel()
            self._calc_acc()
        else:
            self.vel = self.center
            self.acc = self.center
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def _calc_vel(self):
        delta_x = 0
        delta_y = 0
        lim = min(5, len(self.avg_pos) - 1)
        for i in range(lim):
            delta_x += (self.avg_pos[i][0] - self.avg_pos[i + 1][0])
            delta_y += (self.avg_pos[i][1] - self.avg_pos[i + 1][1])
        delta_x /= lim
        delta_y /= lim
        self.vel[0] = delta_x + self.center[0]
        self.vel[1] = delta_y + self.center[1]
        self.vel_q.insert(0, (self.vel[0], self.vel[1], self._scalar_v()))

    def _calc_acc(self):
        delta_x = 0
        delta_y = 0
        lim = min(5, len(self.vel_q) - 1)
        if lim > 0:
            for i in range(lim):
                delta_x += (
                    (self.vel_q[i][0] + 1) * 1000 / (self.avg_pos[i][1] + 10) -
                    (self.vel_q[i + 1][0] + 1) * 1000 / (self.avg_pos[i + 1][1] + 10)
                )
                delta_y += (
                    (self.vel_q[i][1] + 1) * 1000 / (self.avg_pos[i][1] + 10) -
                    (self.vel_q[i + 1][1] + 1) * 1000 / (self.avg_pos[i + 1][1] + 10)
                )
            delta_x /= lim
            delta_y /= lim
            self.acc[0] = delta_x + self.center[0]
            self.acc[1] = delta_y + self.center[1]
            self.acc_q.insert(0, (self.vel[0], self.vel[1], self._scalar_a()))

    def _scalar_v(self):
        return math.sqrt((self.vel[0]) ** 2 + (self.vel[1]) ** 2)

    def _scalar_a(self):
        return math.sqrt((self.acc[0]) ** 2 + (self.acc[1]) ** 2)
